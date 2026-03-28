package hard_problems.reordering;

import java.util.HashSet;
import java.util.Set;

public class ReorderingProblemLoop {
    static int w, x, y, z;

    public static void main(String[] args) throws InterruptedException {
        Set<Answer> answers = new HashSet<>();
        for (int i = 0; i < 100000; i++) {
            w = x = y = z = 0;

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

            answers.add(new Answer(y, z));
        }
        System.out.println("All possible outputs: " + answers);
    }

    static class Answer {
        int r1;
        int r2;

        public Answer(int r1, int r2) {
            this.r1 = r1;
            this.r2 = r2;
        }

        @Override
        public String toString() {
            return "(" + r1 + ", " + r2 + ")";
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Answer answer = (Answer) o;
            return r1 == answer.r1 && r2 == answer.r2;
        }

        @Override
        public int hashCode() {
            return r1 * 31 + r2;
        }
    }

}