package hard_problems.models.deepseek_v3_2;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;

public class NonVolatileReordering {
    static int w = 0, x = 0;
    static int y = 0, z = 0;

    public static void main(String[] args) throws InterruptedException {
        Set<String> outputs = new HashSet<>();
        int iterations = 0;
        
        while (outputs.size() < 4 && iterations < 100000) {
            iterations++;
            w = 0; x = 0; y = 0; z = 0;
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

            outputs.add("(" + y + ", " + z + ")");
        }

        System.out.println("All possible outputs for non-volatile:");
        for (String s : outputs) {
            System.out.println(s);
        }
    }
}
