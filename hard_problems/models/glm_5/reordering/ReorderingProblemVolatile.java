package hard_problems.models.glm_5.reordering;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;

public class ReorderingProblemVolatile {
    // Adding volatile prevents reordering of writes with subsequent reads
    static volatile int w = 0, x = 0;
    static int y, z;

    public static void main(String[] args) throws InterruptedException {
        Set<String> outputs = new HashSet<>();
        boolean foundZeroZero = false;

        for (int i = 0; i < 100_000; i++) {
            w = 0; x = 0;
            
            CountDownLatch latch = new CountDownLatch(2);
            
            Thread t1 = new Thread(() -> {
                latch.countDown();
                try { latch.await(); } catch (InterruptedException e) {}
                w = 1;
                y = x;
            });

            Thread t2 = new Thread(() -> {
                latch.countDown();
                try { latch.await(); } catch (InterruptedException e) {}
                x = 1;
                z = w;
            });

            t1.start();
            t2.start();
            t1.join();
            t2.join();
            
            if (y == 0 && z == 0) {
                foundZeroZero = true;
            }
            outputs.add("(" + y + ", " + z + ")");
        }

        System.out.println("Possible Outputs Found: " + outputs);
        System.out.println("Was (0, 0) found? " + foundZeroZero);
    }
}