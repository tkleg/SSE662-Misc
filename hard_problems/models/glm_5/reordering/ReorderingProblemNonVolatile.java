package hard_problems.models.glm_5.reordering;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;

public class ReorderingProblemNonVolatile {
    static int w = 0, x = 0;
    static int y, z;

    public static void main(String[] args) throws InterruptedException {
        Set<String> outputs = new HashSet<>();
        
        // Run enough times to statistically capture all reorderings
        for (int i = 0; i < 100_000; i++) {
            // Reset shared variables
            w = 0; x = 0;
            
            CountDownLatch latch = new CountDownLatch(2);
            
            Thread t1 = new Thread(() -> {
                latch.countDown();
                try { latch.await(); } catch (InterruptedException e) {}
                w = 1;
                y = x; // Reordering possible: this might happen before w=1
            });

            Thread t2 = new Thread(() -> {
                latch.countDown();
                try { latch.await(); } catch (InterruptedException e) {}
                x = 1;
                z = w; // Reordering possible: this might happen before x=1
            });

            t1.start();
            t2.start();
            t1.join();
            t2.join();
            
            outputs.add("(" + y + ", " + z + ")");
            
            // Optimization: Stop early if we found the rare case
            if (outputs.size() == 4) break; 
        }

        System.out.println("Possible Outputs Found: " + outputs);
    }
}