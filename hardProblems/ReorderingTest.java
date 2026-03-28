package hardProblems;

import java.util.HashSet;
import java.util.Set;

public class ReorderingTest {
    
    public static void main(String[] args) throws InterruptedException {
        Set<Answer> answers = new HashSet<>();
        dfs(0, 0, 0, 0, new boolean[4], answers, new StringBuilder());
        System.out.println("All possible outputs: " + answers);
    }

    // DFS to simulate all possible interleavings
    // opsDone: [t1op1, t1op2, t2op1, t2op2]
    static void dfs(int x, int y, int r1, int r2, boolean[] done, Set<Answer> answers, StringBuilder order) {
        if (done[0] && done[1] && done[2] && done[3]) {
            Answer ans = new Answer(r1, r2);
            answers.add(ans);
            System.out.println("Order: " + order + " -> " + ans);
            return;
        }
        for (int i = 0; i < 4; i++) {
            if (done[i]) 
                continue;

            //Enforce the order of operations within each thread
            if (i == 1 && !done[0])
                continue;
            if (i == 3 && !done[2])
                continue;

            boolean[] nextDone = done.clone();
            nextDone[i] = true;
            int thisX = x, thisY = y, thisR1 = r1, thisR2 = r2;
            String opName = switch (i) {
                case 0 -> "1|1";
                case 1 -> "1|2";
                case 2 -> "2|1";
                case 3 -> "2|2";
                default -> "";
            };
            int len = order.length();
            if (len > 0)
                order.append(" -> ");
            order.append(opName);
            switch (i) {
                case 0 -> thisX = 1;
                case 1 -> thisR1 = thisY;
                case 2 -> thisY = 1;
                case 3 -> thisR2 = thisX;
            }
            dfs(thisX, thisY, thisR1, thisR2, nextDone, answers, order);
            order.setLength(len); //backtrack by undoing changes prior to recursive calls
        }
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