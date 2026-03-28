package hardProblems.Reordering;

import java.util.HashSet;
import java.util.Set;

public class ReorderingProblem {
    static int x = 0, y = 0;
    static int r1 = 0, r2 = 0;

    public static void main(String[] args) throws InterruptedException {
        Set<Answer> answers = new HashSet<>();
        for (int i = 0; i < 100000; i++) {
            x = 0;
            y = 0;
            r1 = 0;
            r2 = 0;
            Thread t1 = new Thread(() -> {
                x = 1;
                r1 = y;
            });

            Thread t2 = new Thread(() -> {
                y = 1;
                r2 = x;
            });

            t1.start();
            t2.start();

            t1.join();
            t2.join();

            answers.add(new Answer(r1, r2));
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