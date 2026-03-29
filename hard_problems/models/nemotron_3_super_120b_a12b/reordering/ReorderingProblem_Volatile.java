package hard_problems.models.nemotron_3_super_120b_a12b.reordering;

import java.util.HashSet;
import java.util.Set;

public class ReorderingProblem_Volatile {
    // volatile fields – give release/acquire semantics
    static volatile int w = 0, x = 0;
    static int y = 0, z = 0;   // y and z are ordinary locals (still shared)

    public static void main(String[] args) throws InterruptedException {
        final int ITERATIONS = 200_000;
        Set<String> seen = new HashSet<>();

        for (int i = 0; i < ITERATIONS; i++) {
            w = x = y = z = 0;          // reset for this run

            Thread t1 = new Thread(() -> {
                w = 1;          // volatile write
                y = x;          // volatile read of x
            });

            Thread t2 = new Thread(() -> {
                x = 1;          // volatile write
                z = w;          // volatile read of w
            });

            t1.start();
            t2.start();
            t1.join();
            t2.join();

            seen.add("(" + y + ", " + z + ")");
        }

        System.out.println("Volatile outputs observed: " + seen);
        // Expected: [(0,1), (1,0), (1,1)]
    }
}