package hard_problems.models.mistral_small_4_119b_2603.reordering;

import java.util.concurrent.CountDownLatch;

public class ReorderingProblemVolatile {
    static volatile int w = 0, x = 0;
    static int y = 0, z = 0;

    public static void main(String[] args) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(2);
        Thread t1 = new Thread(() -> {
            latch.countDown();
            try { latch.await(); } catch(InterruptedException e) {}
            w = 1;
            y = x;
        });
        Thread t2 = new Thread(() -> {
            latch.countDown();
            try { latch.await(); } catch(InterruptedException e) {}
            x = 1;
            z = w;
        });
        t1.start();
        t2.start();
        t1.join();
        t2.join();
        System.out.println("Output: (" + y + ", " + z + ")");
    }
}