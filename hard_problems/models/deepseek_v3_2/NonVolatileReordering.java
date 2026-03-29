package hard_problems.models.deepseek_v3_2;

import java.util.*;

public class NonVolatileReordering {
    static int w = 0, x = 0;
    static int y = 0, z = 0;
    
    public static void main(String[] args) throws InterruptedException {
        Set<String> outputs = new HashSet<>();
        int attempts = 0;
        int maxAttempts = 100000;
        
        // We'll try multiple times to capture all possible interleavings
        while (outputs.size() < 4 && attempts < maxAttempts) {
            attempts++;
            resetVariables();
            
            // Run threads
            Thread t1 = new Thread(() -> {
                w = 1;
                y = x;
            });
            
            Thread t2 = new Thread(() -> {
                x = 1;
                z = w;
            });
            
            t1.start();
            t2.start();
            t1.join();
            t2.join();
            
            outputs.add("(" + y + ", " + z + ")");
        }
        
        System.out.println("All possible outputs for non-volatile w and x:");
        System.out.println("Found " + outputs.size() + " outputs in " + attempts + " attempts:");
        for (String output : outputs) {
            System.out.println(output);
        }
    }
    
    private static void resetVariables() {
        w = 0; x = 0; y = 0; z = 0;
    }
}