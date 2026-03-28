package hardProblems.Reordering;

import hardProblems.Reordering.ReorderingProblemLoop.Answer;
import java.util.HashSet;
import java.util.Set;

public class ReorderingTest {

    public static final int T1_OP1 = 0, T1_OP2 = 1, T2_OP1 = 2, T2_OP2 = 3;
    public static final int ORIG_W = 0, ORIG_X = 0, ORIG_Y = 0, ORIG_Z = 0;

    public static void main(String[] args) throws InterruptedException {
        Set<Answer> answers = new HashSet<>();
        dfs(ORIG_W, ORIG_X, ORIG_Y, ORIG_Z, new boolean[4], answers, new StringBuilder());
        System.out.println("All possible outputs: " + answers);
    }

    static void dfs(int w, int x, int y, int z, boolean[] done, Set<Answer> answers, StringBuilder order) {
        if (done[0] && done[1] && done[2] && done[3]) {
            Answer ans = new Answer(y, z);
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
            int thisW = w, thisX = x, thisY = y, thisZ = z;
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
                    // t1 reads x into y
                    // See current x
                    dfs(thisW, thisX, thisX, thisZ, nextDone, answers, order);
                    // See stale x (original value)
                    dfs(thisW, thisX, ORIG_X, thisZ, nextDone, answers, order);
                }
                case T2_OP2 -> {
                    // t2 reads w into z
                    // See current w
                    dfs(thisW, thisX, thisY, thisW, nextDone, answers, order);
                    // See stale w (original value)
                    dfs(thisW, thisX, thisY, ORIG_W, nextDone, answers, order);
                }
                default -> {
                    // Normal write
                    switch (i) {
                        case T1_OP1 -> thisW = 1;
                        case T2_OP1 -> thisX = 1;
                    }
                    dfs(thisW, thisX, thisY, thisZ, nextDone, answers, order);
                }
            }
            order.setLength(len); //backtrack by undoing changes prior to recursive calls
        }
    }

}