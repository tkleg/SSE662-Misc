package hard_problems.models.falcon3_7b_instruct.reordering;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class VolatileReordering {
    static volatile AtomicInteger w = new AtomicInteger(0);
    static volatile AtomicInteger x = new AtomicInteger(0);
    static volatile AtomicInteger y = new AtomicInteger(0);
    static volatile AtomicInteger z = new AtomicInteger(0);

    public static void main(String[] args) throws InterruptedException {
        Set<Answer> answers = new HashSet<>();
        ExecutorService executor = Executors.newFixedThreadPool(2);

        IntStream.range(0, 1000).forEach(i -> {
            executor.submit(() -> {
                w.set(1);
                y.set(x.get());
            });
            executor.submit(() -> {
                x.set(1);
                z.set(w.get());
            });
            answers.add(new Answer(y.get(), z.get()));
        });

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.MINUTES);
        System.out.println("Volatile Outputs: " + answers);

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