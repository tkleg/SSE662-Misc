package hard_problems.models.kimi_k2_thinking;

import java.util.concurrent.CountDownLatch;

public class NonVolatileReorderingDemo {
    static int w = 0, x = 0;
    static int y = 0, z = 0;

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== Non-Volatile Variables: All Possible Outputs ===");
        
        // Scenario (0, 0): Both threads reorder (read before write)
        demonstrate00();
        
        // Scenario (0, 1): Thread 1 completes before Thread 2
        demonstrate01();
        
        // Scenario (1, 0): Thread 2 completes before Thread 1
        demonstrate10();
        
        // Scenario (1, 1): Interleaved execution
        demonstrate11();
    }

    // (0,0): Force both threads to read BEFORE writing
    static void demonstrate00() throws InterruptedException {
        reset();
        CountDownLatch latch = new CountDownLatch(2);
        
        Thread t1 = new Thread(() -> {
            int local_x = x;  // Read first (simulates reordering)
            latch.countDown();
            try { latch.await(); } catch (Exception e) {}
            w = 1;
            y = local_x;
        });
        
        Thread t2 = new Thread(() -> {
            int local_w = w;  // Read first (simulates reordering)
            latch.countDown();
            try { latch.await(); } catch (Exception e) {}
            x = 1;
            z = local_w;
        });
        
        runThreads(t1, t2);
        System.out.println("Scenario (0,0): (" + y + ", " + z + ")");
    }

    // (0,1): Sequential execution T1 → T2
    static void demonstrate01() throws InterruptedException {
        reset();
        CountDownLatch latch = new CountDownLatch(1);
        
        Thread t1 = new Thread(() -> {
            w = 1;
            y = x;
            latch.countDown();
        });
        
        Thread t2 = new Thread(() -> {
            try { latch.await(); } catch (Exception e) {}
            x = 1;
            z = w;
        });
        
        runThreads(t1, t2);
        System.out.println("Scenario (0,1): (" + y + ", " + z + ")");
    }

    // (1,0): Sequential execution T2 → T1
    static void demonstrate10() throws InterruptedException {
        reset();
        CountDownLatch latch = new CountDownLatch(1);
        
        Thread t1 = new Thread(() -> {
            try { latch.await(); } catch (Exception e) {}
            w = 1;
            y = x;
        });
        
        Thread t2 = new Thread(() -> {
            x = 1;
            z = w;
            latch.countDown();
        });
        
        runThreads(t1, t2);
        System.out.println("Scenario (1,0): (" + y + ", " + z + ")");
    }

    // (1,1): Interleaved (writes complete before reads)
    static void demonstrate11() throws InterruptedException {
        reset();
        CountDownLatch writeLatch = new CountDownLatch(2);
        CountDownLatch readLatch = new CountDownLatch(1);
        
        Thread t1 = new Thread(() -> {
            w = 1;
            writeLatch.countDown();
            try { readLatch.await(); } catch (Exception e) {}
            y = x;
        });
        
        Thread t2 = new Thread(() -> {
            x = 1;
            writeLatch.countDown();
            try { readLatch.await(); } catch (Exception e) {}
            z = w;
        });
        
        t1.start(); t2.start();
        writeLatch.await(); // Wait for both writes
        readLatch.countDown(); // Then release reads
        t1.join(); t2.join();
        System.out.println("Scenario (1,1): (" + y + ", " + z + ")");
    }

    static void reset() { w = 0; x = 0; y = 0; z = 0; }
    static void runThreads(Thread t1, Thread t2) throws InterruptedException {
        t1.start(); t2.start(); t1.join(); t2.join();
    }
}
