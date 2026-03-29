package hard_problems.models.minimax_m2_5.reordering;

import java.util.HashSet;
import java.util.Set;

public class ReorderingNonVolatile {
    // non‑volatile shared variables
    static int w = 0, x = 0;
    // results of the two threads
    static int y = 0, z = 0;

    public static void main(String[] args) throws InterruptedException {
        Set<String> observed = new HashSet<>();

        // Keep trying until we have seen the four different results
        for (int i = 1; ; i++) {
            // reset the shared fields before each iteration
            w = 0; x = 0; y = 0; z = 0;

            // latch makes the two threads start as close as possible
            java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(2);

            Thread t1 = new Thread(() -> {
                latch.countDown();
                try { latch.await(); } catch (InterruptedException e) { }

                w = 1;            // write to w
                y = x;            // read x
            });

            Thread t2 = new Thread(() -> {
                latch.countDown();
                try { latch.await(); } catch (InterruptedException e) { }

                x = 1;            // write to x
                z = w;            // read w
            });

            t1.start();
            t2.start();
            t1.join();
            t2.join();

            String result = "(" + y + ", " + z + ")";
            observed.add(result);

            System.out.println("Iteration " + i + ": " + result);

            // As soon as we have all four possibilities we can stop.
            if (observed.size() == 4) {
                System.out.println("All four outputs observed after " + i + " iterations.");
                break;
            }
        }
        System.out.println("Possible outputs (non-volatile): " + observed);
    }
}