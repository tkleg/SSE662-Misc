package hard_problems.map_tricks;

import java.util.*;

class Key {
    int val;
    Key(int val) { this.val = val; }
    @Override
    public int hashCode() { return val; }
    @Override
    public boolean equals(Object o) {
        return o instanceof Key && ((Key)o).val == this.val;
    }
    @Override
    public String toString() { return String.valueOf(val); }
}

public class MapTricks {
    public static void main(String[] args) {
        Map<Key, String> hashMap = new HashMap<>();
        Key k1 = new Key(1);
        Key k2 = new Key(2);
        Key k3 = new Key(3);

        hashMap.put(k1, "one");
        hashMap.put(k2, "two");
        hashMap.put(k3, "three");

        k1.val = 3;
        k2.val = 1;

        hashMap.computeIfAbsent(new Key(1), key -> hashMap.getOrDefault(new Key(2), "X"));
        hashMap.computeIfAbsent(new Key(3), key -> hashMap.getOrDefault(new Key(1), "Y"));

        TreeMap<String, Integer> treeMap = new TreeMap<>((a, b) -> a.length() - b.length());

        treeMap.put("aa", 1);
        treeMap.put("bb", 2);
        treeMap.put("c", 3);
        treeMap.merge("dd", 4, Integer::sum);
        treeMap.merge("e", 5, Integer::sum);
        treeMap.merge("fff", 6, Integer::sum);

        LinkedHashMap<Integer, String> linkedMap = new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Integer,String> eldest) {
                return size() > 3;
            }
        };
        linkedMap.put(1, "A");
        linkedMap.put(2, "B");
        linkedMap.put(3, "C");
        linkedMap.get(1);
        linkedMap.put(4, "D");
        linkedMap.get(2);
        linkedMap.put(5, "E");

        Map<Integer, String> identityMap = new IdentityHashMap<>();
        Integer a = 127, b = 127, c = 128, d = 128;
        identityMap.put(a, "X");
        identityMap.put(b, "Y");
        identityMap.put(c, "Z");
        identityMap.put(d, "W");

        System.out.println("HashMap:");
        hashMap.forEach((k,v) -> System.out.print(k + ":" + v + " "));
        System.out.println("\nTreeMap:");
        treeMap.forEach((k,v) -> System.out.print(k + ":" + v + " "));
        System.out.println("\nLinkedHashMap:");
        linkedMap.forEach((k,v) -> System.out.print(k + ":" + v + " "));
        System.out.println("\nIdentityHashMap:");
        identityMap.forEach((k,v) -> System.out.print(k + ":" + v + " "));
    }
}