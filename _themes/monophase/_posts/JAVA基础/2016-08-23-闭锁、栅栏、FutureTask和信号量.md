---
layout: post
comments: true
categories: java基础
---

&emsp;&emsp;同步工具类可以是任何一个对象，只要它根据自身的状态来协调线程的控制流。阻塞队列可以作为同步类工具，其他类型的同步工具类还包括信号量、闭锁、栅栏和FutureTask。所有的同步工具类都包含一些特定的结构化属性：他们封装了一些状态，这些状态将决定执行同步工具类的线程是继续执行还是等待，此外还提供了一些方法对状态进行操作，以及另一些方法用于高效地等待同步工具类进入语气的状态。  

### 一、闭锁
&emsp;&emsp;闭锁是一种同步工具类，可以控制线程的进度，闭锁的作用相当于一扇门：在闭锁到达结束状态之前，这扇门一直是关着的，并且没有任何线程可以通过，当达到结束状态时，这扇门会自动打开并允许所有线程通过。这扇门一旦打开，就不会再关闭。  

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

&emsp;&emsp;这段代码的作用是统计所有的线程同时开始执行某个任务到最后一个线程完成任务的时间。为了保证所有的线程同时开始执行任务，使用了startGate。当所有线程执行完，endGate会停止等待。

### 二、栅栏
&emsp;&emsp;栅栏类似于闭锁，它能阻塞一组线程直到某个事件发生。栅栏与闭锁的关键区别在于，所有线程必须同时到达栅栏位置才能继续执行。闭锁用于等待事件，而栅栏用于等待其他线程。闭锁一旦打开，就不能被重置，而栅栏自动重置。下面的代码是通过CyclicBarrier协调细胞自动衍生系统中的计算。

```
public class CellularAutomata {
	private final Board mainBoard;
	private final CyclicBarrier barrier;
	private final Worker[] workers;

	public CellularAutomata(Board board) {
		this.mainBoard = board;
		int count = Runtime.getRuntime().availableProcessors();
		this.barrier = new CyclicBarrier(count,
			new Runnable() {
				@Override
				public void run() {
					mainBoard.commitNewValues();
				}});
		this.workers = new Worker[count];
		for(int i = 0; i < count; i++) {
			workers[i] = new Worker(mainBoard.getSubBoard(count,i));
		}

	}

	private class Worker implements Runnable {
		private final Board board;
		public Worker(Board board){
			this.board = board;
		}
		public void run() {
			while(!board.hasConverged()) {
				for (int x = 0; x < board.getMaxX(); x++){
					for(int y = 0; y < board.getMaxY(); y++){
						board.setNewValue(x,y,computeValue(x,y));
					}
				}
				try {
					barrier.await();
				} catch(InterruptedException ex){
					return;
				} catch (BrokenBarrierException ex){
					return;
				}
			}
		}
	}
}
```

&emsp;&emsp;上面的例子是java并发编程实战中给出的例子，第一次看有点眼晕，我试着写了个好理解的版本。程序描述的情景是这样的。A、B、C、D四个人一起跑步，都要跑五圈，每次跑完一圈都要等所有人到达终点再继续跑。 又不是比赛，被套圈多尴尬呀。

```
public class RunGame {

	public static void main(String[] args) {
		 CyclicBarrier barrier = new CyclicBarrier(4, new Runnable() {
				@Override
				public void run() {
					System.out.println("all start run");
				}});
		 Runner A = new Runner(barrier, "A");
		 Runner B = new Runner(barrier, "B");
		 Runner C = new Runner(barrier, "C");
		 Runner D = new Runner(barrier, "D");
		 new Thread(A).start();
		 new Thread(B).start();
		 new Thread(C).start();
		 new Thread(D).start();

	}
}

class Runner implements Runnable{

	private final String name;
	private final CyclicBarrier barrier;

	public Runner(CyclicBarrier barrier, String name) {
		this.barrier = barrier;
		this.name = name;
	}

	@Override
	public void run() {
		for(int i = 1; i <= 5; i++)
		try{
			barrier.await();
			System.out.println(name + " start run " + i + " cycle");
		} catch(Exception e){
			e.printStackTrace();
		};
	}
}

```
&emsp;&emsp;上面的例子简单明了多了，我们看看输出结果：

```
all start run
A start run 1 cycle
D start run 1 cycle
B start run 1 cycle
C start run 1 cycle
all start run
C start run 2 cycle
A start run 2 cycle
B start run 2 cycle
D start run 2 cycle
all start run
D start run 3 cycle
C start run 3 cycle
B start run 3 cycle
A start run 3 cycle
all start run
A start run 4 cycle
C start run 4 cycle
D start run 4 cycle
B start run 4 cycle
all start run
B start run 5 cycle
A start run 5 cycle
C start run 5 cycle
D start run 5 cycle
```
&emsp;&emsp;从输出结果可以看到，大家都跑完了五圈，而且没有套圈的情况发生。   

### 三、信号量
&emsp;&emsp;计数信号量用来控制同时访问某个特定资源的操作数量，或者同时执行某个指定操作的数量。技术信号量还可以用来实现某种资源池，或者对容器施加边界。   
&emsp;&emsp;Semaphore中管理着一组虚拟的许可，许可的初始数量可以通过构造函数来指定。在执行操作时可以首先获取许可，并在得到许可并执行后释放许可。如果没有许可，那么acquire将阻塞直到有许可为止。   

```
public class BoundedHashSet<T> {
	private final Set<T> set;
	private final Semaphore sem;

	public BoundedHashSet(int bound) {
		this.set = Collections.synchronizedSet(new HashSet<T>());
		sem = new Semaphore(bound);
	}

	public boolean add(T o) throws InterruptedException {
		sem.acquire();
		boolean wasAdded = false;
		try {
			wasAdded = set.add(o);
			return wasAdded;
		} finally {
			if (!wasAdded) {
				sem.release();
			}
		}
	}

	public boolean remove(Object o) {
		boolean wasRemoved = set.remove(o);
		if (wasRemoved) {
			sem.release();
		}
		return wasRemoved;
	}
}
```

### 四、FutureTask
&emsp;&emsp;FutureTask也可以用做闭锁。FutureTask表示的是计算是通过Callable实现的，相当于一种可生成结果的Runnable，并且可以处于一下三种状态：等待状态、正在运行和运行完成。Future.get的行为取决于任务的状态。如果任务已经完成，那么get会立刻返回结果，否则get将阻塞直到任务进入完成状态，然后返回结果或者抛出异常。FutureTask将计算结果从执行计算的线程传递到获取这个结果的线程，而FutureTask的规范确保了这种传递过程能实现结果。  
&emsp;&emsp;FutureTask在Executor框架中表示异步任务，此外还可以用来表示一些时间较长的计算，这些计算可以在使用计算结果之前启动。

```
public class CalculateSum {
	public static final FutureTask<Integer> future = new FutureTask<>(new Callable<Integer>(){
		@Override
		public Integer call() throws Exception {
			return 3+4;
		}

	});
	public static void main(String[] args) throws InterruptedException, ExecutionException {
		new Thread(future).start();
		System.out.println(future.get());
	}
}
```
