package hard_problems.string_interning;

public class StringInterningScrambled {
    @SuppressWarnings({"StringEquality", "RedundantStringConstructorCall"})
    public static void main(String[] args) {
        String a = "Hello";
        String b = "Hello";
        String c = a + "";
        String d = new String("Hello");
        String e = c.intern();


        String f = "foo";
        String g = f + "bar";
        String h = "foobar";

        String i = "fo" + "obar";

        String j = new String("foobar");
        String jj = j.substring(0);
        String jjj = j.subSequence(0, 6).toString();

        String k = new String("foobar").intern();

        String l = "fo" + new String("obar");

        String m = "𝔘𝔫𝔦𝔠𝔬𝔡𝔢";
        String n = ("𝔘" + "𝔫" + "𝔦" + "𝔠" + "𝔬" + "𝔡" + "𝔢");
        String o = new String("𝔘𝔫𝔦𝔠𝔬𝔡𝔢");

        // Final variables and constants
        final String p = "baz";
        String q = p + "qux";
        String r = "bazqux";

        String s = ("ba" + "z") + ("qu" + "x");

        String t = (p + new String("qux"));
         System.out.println("k == h: " + (k == h));
        System.out.println("j == h: " + (j == h));
        System.out.println("j == j.intern(): " + (j == j.intern()));
        System.out.println("t.intern() == r: " + (t.intern() == r));
        System.out.println("t == r: " + (t == r));
        System.out.println("s == r: " + (s == r));
        System.out.println("g == h: " + (g == h));
        System.out.println("j.intern() == h: " + (j.intern() == h));
        System.out.println("m == o: " + (m == o));
        System.out.println("a == c: " + (a == c));
        System.out.println("a == d: " + (a == d));
        System.out.println("l == h: " + (l == h));
        System.out.println("a == e: " + (a == e));
        System.out.println("m == n: " + (m == n));
        System.out.println("i == h: " + (i == h));
        System.out.println("a == b: " + (a == b));
        System.out.println("f == g: " + (f == g));
        System.err.println("jjj == h: " + (jjj == h));
        System.out.println("l.intern() == h: " + (l.intern() == h));
        System.out.println("g.intern() == h: " + (g.intern() == h));
        System.out.println("q == r: " + (q == r));
        System.out.println("o.intern() == m: " + (o.intern() == m));
        System.err.println("jj == h: " + (jj == h));
     }
}
