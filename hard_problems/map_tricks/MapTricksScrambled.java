package hard_problems.map_tricks;

import java.util.*;

class Key {
    int val;
    Key(int val){
        this.val = val;
    }

    @Override
    public int hashCode(){
        return val;
    }

    @Override
    public boolean equals(Object o) {
        if( !( o instanceof Key ) ) 
            return false;
        Key other = (Key) o;
        return this.val == other.val;
    }

    @Override
    public String toString(){
        return ""+val;
    }

}

public class MapTricksScrambled {
    public static void main(String[] args) {
        Map<Key, String> hashMap = new HashMap<>();
        LinkedHashMap<Integer, String> linkedMap = new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Integer,String> eldest) {
                return size() > 3;
            }
        };
        Key k1 = new Key(1);
        TreeMap<String, Integer> treeMap = new TreeMap<>((a, b) -> a.length() - b.length());
        Key k2 = new Key(2);
        Map<Integer, String> identityMap = new IdentityHashMap<>();
        Key k3 = new Key(3);
        Integer a = 127, b = 127, c = 128, d = 128;

        hashMap.put(k1, "one");
        hashMap.put(k2, "two");
        hashMap.put(k3, "three");

        k1.val = 3;
        k2.val = 1;

        hashMap.computeIfAbsent(new Key(1), key -> hashMap.getOrDefault(new Key(2), "X"));
        hashMap.computeIfAbsent(new Key(3), key -> hashMap.getOrDefault(new Key(1), "Y"));

        linkedMap.put(4, "D");
        linkedMap.put(3, "C");
        identityMap.put(b, "Y");
        linkedMap.get(2);
        linkedMap.get(1);
        treeMap.merge("dd", 4, Integer::sum);
        linkedMap.put(1, "A");
        linkedMap.put(2, "B");
        treeMap.merge("fff", 6, Integer::sum);
        treeMap.put("c", 3);
        identityMap.put(a, "X");
        treeMap.put("aa", 1);
        identityMap.put(c, "Z");
        treeMap.put("bb", 2);
        linkedMap.put(5, "E");
        treeMap.merge("e", 5, Integer::sum);
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