package hard_problems.models.nemotron_3_super_120b_a12b;

import java.util.HashSet;
import java.util.Set;

public class ReorderingProblem_NonVolatile {
    // plain fields – no synchronization
    static int w = 0, x = 0;
    static int y = 0, z = 0;

    public static void main(String[] args) throws InterruptedException {
        final int ITERATIONS = 200_000;          // enough to see all four results
        Set<String> seen = new HashSet<>();

        for (int i = 0; i < ITERATIONS; i++) {
            // reset the shared variables for this iteration
            w = x = y = z = 0;

            Thread t1 = new Thread(() -> {
                w = 1;          // write w
                y = x;          // read x
            });

            Thread t2 = new Thread(() -> {
                x = 1;          // write x
                z = w;          // read w
            });

            t1.start();
            t2.start();
            t1.join();
            t2.join();

            seen.add("(" + y + ", " + z + ")");
        }

        System.out.println("Non‑volatile outputs observed: " + seen);
        // Expected: [(0,0), (0,1), (1,0), (1,1)]
    }
}