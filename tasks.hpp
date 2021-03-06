#pragma once
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#include <pthread.h>

#include <unistd.h>
#include "util.h"

constexpr int CPU_DEVICE = -1;
constexpr int NULL_EVENT = 0;

typedef void (*TaskFunc) (int, void*);

struct Task {
    int event_id;
    int wait_on;

    TaskFunc task_func;
    void* user_data;

    const char* name;
};

enum EventType {
    ET_SIMPLE_EVENT,
    ET_AGGREGATE_EVENT
};

struct SimpleEvent {
    bool done = false;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
};

struct AggregateEvent {
    int num_events;
    int* event_list;
};

struct Event {
    Event() {
        type = ET_SIMPLE_EVENT;
        simple.done = false;
        pthread_mutex_init(&simple.mutex, NULL);
        pthread_cond_init(&simple.cond, NULL);
    }

    EventType type;

    union {
        SimpleEvent simple;
        AggregateEvent aggregate;
    };
};

struct GpuTaskScheduler;

struct TaskRunnerArgs {
    int device_id;
    GpuTaskScheduler* scheduler;
};

struct Duration {
    int device_id;
    const char* name;
    double start_time, stop_time;
};

struct GpuTaskScheduler {
    std::vector<Task> cpu_queue;

    std::vector<std::vector<Task>> gpu_queues;

    std::vector<Event> events;
    std::vector<Duration> durations;

    double start_time;

    cudaStream_t* streams;

    GpuTaskScheduler(int ngpus, cudaStream_t* streams) : streams(streams) {
        gpu_queues.resize(ngpus);

        Event null_event = {}; 
        null_event.type = ET_SIMPLE_EVENT;
        null_event.simple.done = true;

        events.push_back(null_event);
    }

    template <typename TaskArgs>
    int enqueue_task(const char* name, int device_id, int event_id, TaskFunc task_func, TaskArgs* args) {
        Task t = {};
        t.name = name;
        t.task_func = task_func;

        // Copy the arguments...
        t.user_data = malloc(sizeof(TaskArgs));
        memcpy(t.user_data, args, sizeof(TaskArgs));

        t.wait_on = event_id;

        Event event;
        event.type = ET_SIMPLE_EVENT;
        event.simple.done = false;
        event.simple.mutex = PTHREAD_MUTEX_INITIALIZER;
        event.simple.cond  = PTHREAD_COND_INITIALIZER;

        events.push_back(event);
        t.event_id = events.size() - 1;

        if (device_id == -1)
            cpu_queue.push_back(t);
        else
            gpu_queues[device_id].push_back(t);

        return t.event_id;
    }

    int aggregate_event(const std::vector<int>& event_list) {
        Event event;
        event.type = ET_AGGREGATE_EVENT;

        event.aggregate.num_events = event_list.size();
        event.aggregate.event_list = (int*)malloc(sizeof(int) * event_list.size());

        for (size_t i = 0; i < event_list.size(); i++)
            event.aggregate.event_list[i] = event_list[i];

        events.push_back(event);
        return events.size()-1;
    }

    int ngpus() {
        return gpu_queues.size();
    }

    // Runs all scheduled tasks and wait for completion.
    void run() {
        std::vector<pthread_t> threads(ngpus());
        std::vector<TaskRunnerArgs> args(ngpus());

        start_time = get_real_time();
        durations.resize(events.size());

        for (int i = 0; i < ngpus(); i++) {
            args[i].scheduler = this;
            args[i].device_id = i;
            pthread_create(&threads[i], nullptr, run_all_tasks, &args[i]);
        }

        for (size_t i = 0; i < cpu_queue.size(); i++) {
            run_task(-1, &cpu_queue[i]);
        }

        for (int i = 0; i < ngpus(); i++) {
            pthread_join(threads[i], nullptr);
        }
    }

private:
    static void* run_all_tasks(void* p) {
        TaskRunnerArgs* args = (TaskRunnerArgs*)p;
        GpuTaskScheduler* scheduler = args->scheduler;
        int device_id = args->device_id;

        std::vector<Task>& my_queue = scheduler->gpu_queues[device_id];        

        CUDA_CALL(cudaSetDevice(device_id));

#ifdef PROFILE_TASKS
        std::vector<cudaEvent_t> start_events(my_queue.size()), end_events(my_queue.size());
        for (size_t i = 0; i < my_queue.size(); i++) {
            CUDA_CALL(cudaEventCreate(&start_events[i]));
            CUDA_CALL(cudaEventCreate(&end_events[i]));
        }

        cudaEvent_t scheduler_start;
        CUDA_CALL(cudaEventCreate(&scheduler_start));
        CUDA_CALL(cudaEventRecord(scheduler_start, scheduler->streams[device_id]));
#endif

        for (size_t i = 0; i < my_queue.size(); i++) {
            Task* t = &my_queue[i];

#ifdef PROFILE_TASKS
            CUDA_CALL(cudaEventRecord(start_events[i], scheduler->streams[device_id]));
#endif
            scheduler->run_task(device_id, t);
#ifdef PROFILE_TASKS
            CUDA_CALL(cudaEventRecord(end_events[i], scheduler->streams[device_id]));
#endif
        }

        CUDA_CALL(cudaSetDevice(device_id));
        CUDA_CALL(cudaDeviceSynchronize());

#ifdef PROFILE_TASKS
        for (size_t i = 0; i < my_queue.size(); i++) {
            float start_elapsed, end_elapsed;

            CUDA_CALL(cudaEventElapsedTime(&start_elapsed, scheduler_start, start_events[i]));
            CUDA_CALL(cudaEventElapsedTime(&end_elapsed, scheduler_start, end_events[i]));

            scheduler->durations[my_queue[i].event_id].device_id = device_id;
            scheduler->durations[my_queue[i].event_id].name = my_queue[i].name;
            scheduler->durations[my_queue[i].event_id].start_time = start_elapsed/1000.0;
            scheduler->durations[my_queue[i].event_id].stop_time = end_elapsed/1000.0;
        }
#endif

        return nullptr;
    }

    void wait_for_event(int event_id) {
        if (events[event_id].type == ET_SIMPLE_EVENT) {
            if (events[event_id].simple.done)
                return;

            pthread_mutex_lock(&events[event_id].simple.mutex);
            while (!events[event_id].simple.done) {
                pthread_cond_wait(&events[event_id].simple.cond, &events[event_id].simple.mutex);
            }
            pthread_mutex_unlock(&events[event_id].simple.mutex);
        } else {
            for (int i = 0; i < events[event_id].aggregate.num_events; i++) {
                wait_for_event(events[event_id].aggregate.event_list[i]);
            }
        }
    }

    void run_task(int device, Task* task) {
        wait_for_event(task->wait_on);

#ifdef DEBUG_TASKS
        printf("D(%d): Starting task %s(%d)\n", device, task->name, task->event_id);
        fflush(stdout);
#endif

#ifdef PROFILE_TASKS
        double t_begin = get_real_time();
#endif
        task->task_func(device, task->user_data);

#ifdef PROFILE_TASKS
        double t_end = get_real_time();
        durations[task->event_id].device_id = -1;
        durations[task->event_id].name = task->name;
        durations[task->event_id].start_time = t_begin - start_time;
        durations[task->event_id].stop_time  = t_end - start_time;
#endif 

#ifdef DEBUG_TASKS
        printf("D(%d): Stop task %s(%d)\n", device, task->name, task->event_id);
        fflush(stdout);
#endif

        if (events[task->event_id].type == ET_SIMPLE_EVENT) {
            pthread_mutex_lock(&events[task->event_id].simple.mutex);
            events[task->event_id].simple.done = true;
            pthread_cond_broadcast(&events[task->event_id].simple.cond);
            pthread_mutex_unlock(&events[task->event_id].simple.mutex);
        }
    }
};
