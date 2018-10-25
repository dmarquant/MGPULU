#pragma once
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#include <pthread.h>

#include <unistd.h>

typedef void (*TaskFunc) (int, void*);

struct Task {
    int event_id;
    int wait_on;

    TaskFunc task_func;
    void* user_data;
};

struct Event {
    bool done = false;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
};

struct GpuTaskScheduler;

struct TaskRunnerArgs {
    int device_id;
    GpuTaskScheduler* scheduler;
};

struct GpuTaskScheduler {
    std::vector<Task> cpu_queue;

    std::vector<std::vector<Task>> gpu_queues;

    std::vector<Event> events;

public:
    GpuTaskScheduler(int ngpus) {
        gpu_queues.resize(ngpus);

        Event null_event = {};
        null_event.done = true;

        events.push_back(null_event);
    }

    template <typename TaskArgs>
    int enqueue_task(int device_id, int event_id, TaskFunc task_func, TaskArgs* args) {
        Task t = {};
        t.task_func = task_func;

        // Copy the arguments...
        t.user_data = malloc(sizeof(TaskArgs));
        memcpy(t.user_data, args, sizeof(TaskArgs));

        t.wait_on = event_id;

        events.emplace_back();
        t.event_id = events.size() - 1;

        if (device_id == -1)
            cpu_queue.push_back(t);
        else
            gpu_queues[device_id].push_back(t);

        return t.event_id;
    }

    int ngpus() {
        return gpu_queues.size();
    }

    // Runs all scheduled tasks and wait for completion.
    void run() {
        std::vector<pthread_t> threads(ngpus());
        std::vector<TaskRunnerArgs> args(ngpus());

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

        for (size_t i = 0; i < my_queue.size(); i++) {
            Task* t = &my_queue[i];
            scheduler->run_task(device_id, t);
        }

        return nullptr;
    }

    void wait_for_event(int event_id) {
        if (events[event_id].done)
            return;

        pthread_mutex_lock(&events[event_id].mutex);
        while (!events[event_id].done) {
            pthread_cond_wait(&events[event_id].cond, &events[event_id].mutex);
        }
        pthread_mutex_unlock(&events[event_id].mutex);
    }

    void run_task(int device, Task* task) {
        wait_for_event(task->wait_on);

        task->task_func(device, task->user_data);

        pthread_mutex_lock(&events[task->event_id].mutex);
        events[task->event_id].done = true;
        pthread_cond_broadcast(&events[task->event_id].cond);
        pthread_mutex_unlock(&events[task->event_id].mutex);
    }
};

struct SleepArgs {
    int sec;
};

void sleep_func(int device, void* p) {
    SleepArgs* args = (SleepArgs*)p;

    printf("Device(%d): Going to sleep for %d seconds\n", device, args->sec);
    sleep(args->sec);
    printf("Device(%d): Slept for %d seconds\n", device, args->sec);
}
