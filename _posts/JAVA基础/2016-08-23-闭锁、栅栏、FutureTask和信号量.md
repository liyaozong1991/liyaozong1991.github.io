---
layout: post
comments: true
categories: java基础
---
## java同步工具类——闭锁、栅栏、FutureTask和信号量
同步工具类可以是任何一个对象，只要它根据自身的状态来协调线程的控制流。阻塞队列可以作为同步类工具，
其他类型的同步工具类还包括信号量、闭锁、栅栏和FutureTask。所有的同步工具类都包含一些特定的结构化属性：
他们封装了一些状态，这些状态将决定执行同步工具类的线程是继续执行还是等待，此外还提供了一些方法对状态进行操作，
以及另一些方法用于高效地等待同步工具类进入语气的状态。  
### 一、闭锁
闭锁是一种同步工具类，可以控制线程的进度，闭锁的作用相当于一扇门：在闭锁到达结束状态之前，这扇门一直是关着的，
并且没有任何线程可以通过，当达到结束状态时，这扇门会自动打开并允许所有线程通过。这扇门一旦打开，就不会再关闭。
```
public class TestHarness {
    public long timeTasks(int nThreads, final Runnable task) throws InterruptedException {
        final CountDownLatch startGate = new CountDownLatch(1);
        final CountDownLatch endGate = new CountDownLatch(nThreads);
        for (int i = 0; i < nThreads; i++) {
            Thread t = new Thread() {
                public void run() {
                    try {
                        startGate.await();
                        try {  
                            task.run();
                        } finally {
                            endGate.countDown();
                        }
                    } catch (InterruptedException ignored) {}
                }
            };
        }
        long start = System.nanoTime();
        startGate.countDown();
        endGate.await();
        long end = System.nanoTime();
        return end - start;
    }
}
```

这段代码的作用是统计所有的线程同时开始执行某个任务到最后一个线程完成任务的时间。为了保证所有的线程同时开始执行任务，
使用了startGate。当所有线程执行完，endGate会停止等待。

### 二、栅栏
栅栏类似于闭锁，它能阻塞一组线程直到某个事件发生。栅栏与闭锁的关键区别在于，所有线程必须同时到达栅栏位置才能继续执行。
闭锁用于等待事件，而栅栏用于等待其他线程。闭锁一旦打开，就不能被重置，而栅栏自动重置。

