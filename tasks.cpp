#include "tasks.hpp"

int main() {
    GpuTaskScheduler scheduler(2);

    SleepArgs s1{2};
    SleepArgs s2{2};

    SleepArgs g1{4};
    SleepArgs g2{3};

    int cpu_sleep = scheduler.enqueue_task(-1, 0, sleep_func, &s1);
    int gpu_sleep = scheduler.enqueue_task(0, 0, sleep_func, &g1);
    scheduler.enqueue_task(1, cpu_sleep, sleep_func, &g2);
    scheduler.enqueue_task(-1, gpu_sleep, sleep_func, &s2);

    scheduler.run();

    return 0;
}
