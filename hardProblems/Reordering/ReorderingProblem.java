package hardProblems.Reordering;

public class ReorderingProblem {
    static int w = 0, x = 0;
    static int y = 0, z = 0;

    public static void main(String[] args) throws InterruptedException {
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

        System.out.println("Output: (" + y + ", " + z + ")");
    }

}