package hardProblems.Reordering;

import hardProblems.Reordering.ReorderingProblem.Answer;
import java.util.HashSet;
import java.util.Set;

public class ReorderingTest {

    public static final int T1_OP1 = 0, T1_OP2 = 1, T2_OP1 = 2, T2_OP2 = 3;
    public static final int ORIG_X = 0, ORIG_Y = 0, ORIG_R1 = 0, ORIG_R2 = 0;

    public static void main(String[] args) throws InterruptedException {
        Set<Answer> answers = new HashSet<>();
        dfs(ORIG_X, ORIG_Y, ORIG_R1, ORIG_R2, new boolean[4], answers, new StringBuilder());
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
        for (int i = T1_OP1; i <= T2_OP2; i++) {
            if (done[i])
                continue;

            //Enforce the order of operations within each thread
            if (i == T1_OP2 && !done[T1_OP1])
                continue;
            if (i == T2_OP2 && !done[T2_OP1])
                continue;

            boolean[] nextDone = done.clone();
            nextDone[i] = true;
            int thisX = x, thisY = y, thisR1 = r1, thisR2 = r2;
            String opName = switch (i) {
                case T1_OP1 -> "1|1";
                case T1_OP2 -> "1|2";
                case T2_OP1 -> "2|1";
                case T2_OP2 -> "2|2";
                default -> "";
            };
            int len = order.length();
            if (len > 0)
                order.append(" -> ");
            order.append(opName);
            switch (i) {
                case T1_OP2 -> {
                    //t1 reads y
                    //Try both: see current y, or see stale Y
                    //Operation : r1 = y

                    //See current y
                    dfs(thisX, thisY, thisY, thisR2, nextDone, answers, order);
                    //See stale Y
                    dfs(thisX, thisY, ORIG_Y, thisR2, nextDone, answers, order);
                }
                case T2_OP2 -> {
                    //t2 reads x
                    //Try both: see current x, or see stale X
                    //Operation : r2 = x

                    //See current x
                    dfs(thisX, thisY, thisR1, thisX, nextDone, answers, order);
                    //See stale X
                    dfs(thisX, thisY, thisR1, ORIG_X, nextDone, answers, order);
                }
                default -> {
                    //Normal write, these can only be executed once, so no need to try multiple values
                    switch (i) {
                        case T1_OP1 -> thisX = 1;
                        case T2_OP1 -> thisY = 1;
                    }   dfs(thisX, thisY, thisR1, thisR2, nextDone, answers, order);
                }
            }
            order.setLength(len); //backtrack by undoing changes prior to recursive calls
        }
    }

}