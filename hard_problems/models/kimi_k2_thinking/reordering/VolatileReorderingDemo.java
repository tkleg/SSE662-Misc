package hard_problems.models.kimi_k2_thinking.reordering;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;

public class VolatileReorderingDemo {
    static volatile int w = 0, x = 0;
    static int y = 0, z = 0;

    public static void main(String[] args) throws InterruptedException {
        Set<String> seen = new HashSet<>();
        final int maxIterations = 1_000_000;

        for (int i = 0; i < maxIterations; i++) {
            w = 0; x = 0; y = 0; z = 0;

            CountDownLatch latch = new CountDownLatch(2);

            Thread t1 = new Thread(() -> {
                latch.countDown();
                try { latch.await(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                w = 1;          // volatile write
                y = x;          // volatile read of x
            });

            Thread t2 = new Thread(() -> {
                latch.countDown();
                try { latch.await(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                x = 1;          // volatile write
                z = w;          // volatile read of w
            });

            t1.start();
            t2.start();
            t1.join();
            t2.join();

            seen.add("(" + y + ", " + z + ")");
            if (seen.size() == 3) {
                System.out.println("All three outputs observed after " + (i + 1) + " iterations.");
                break;
            }
        }
        System.out.println("Possible outputs (volatile): " + seen);
    }
}