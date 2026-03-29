package hard_problems.models.qwen3_5_122b_a10b.reordering;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;

public class ReorderingNonVolatile {
    // Non-volatile static variables
    static int w = 0, x = 0;
    static int y = 0, z = 0;

    public static void main(String[] args) throws InterruptedException {
        Set<String> observedOutputs = new HashSet<>();
        
        // Loop until we have observed all 4 possible combinations
        while (observedOutputs.size() < 4) {
            resetVariables();
            
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
            
            String output = "(" + y + ", " + z + ")";
            observedOutputs.add(output);
        }
        
        System.out.println("All possible outputs observed: " + observedOutputs);
    }
    
    private static void resetVariables() {
        w = 0; x = 0; y = 0; z = 0;
    }
}