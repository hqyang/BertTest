--- binarize.py	(original)
+++ binarize.py	(refactored)
@@ -255,7 +255,7 @@
             break
       else:
          if self.log != None:
-            print >> self.log, "can't find a rule for " + sLabel + " with " + ", ".join([child_node.name for child_node in srcnode.children])
+            print("can't find a rule for " + sLabel + " with " + ", ".join([child_node.name for child_node in srcnode.children]), file=self.log)
          nIndex = -1
          headchild = srcnode.children[nIndex]
          while self.not_empty(headchild) == False:
@@ -355,7 +355,7 @@
                   node.head_child = 'l'
                else:
                   if headchild != tuplechildren[1][0]:
-                     print headchild, tuplechildren[0][0], tuplechildren[1][0]
+                     print(headchild, tuplechildren[0][0], tuplechildren[1][0])
                   assert headchild == tuplechildren[1][0]
                   node.head_child = 'r'
                return True
@@ -414,11 +414,11 @@
             outh = CBinarizedTreeNode()
       #      print outh
             self.build_binarized_node(outh, head)
-            print outh
+            print(outh)
          else:
             outh = fidtree.CTreeNode()
             self.build_node(outh, head)
-            print outh
+            print(outh)
 
    def process_noroot(self, sSentence, wfile):
       # don't process empty sentences
@@ -453,17 +453,17 @@
    try:
       opts, args = getopt.getopt(sys.argv[1:], "nuel:d:")
    except getopt.GetoptError: 
-      print "\nUsage: binarize.py [-nue] [-llogfile] [-ddictionary_file] rule_file input_file > output\n"
-      print "-n: not binarize\n"
-      print "-u: remove unary nodes\n"
-      print "-d: replace with dictionary\n"
+      print("\nUsage: binarize.py [-nue] [-llogfile] [-ddictionary_file] rule_file input_file > output\n")
+      print("-n: not binarize\n")
+      print("-u: remove unary nodes\n")
+      print("-d: replace with dictionary\n")
       sys.exit(1)
    if len(args) != 2:
-      print "\nUsage: binarize.py [-nu] [-llogfile] rule_file input_file > output\n"
-      print "-n: not binarize\n"
-      print "-u: remove unary nodes\n"
-      print "-e: keep empty nodes\n"
-      print "-d: replace with dictionary\n"
+      print("\nUsage: binarize.py [-nu] [-llogfile] rule_file input_file > output\n")
+      print("-n: not binarize\n")
+      print("-u: remove unary nodes\n")
+      print("-e: keep empty nodes\n")
+      print("-d: replace with dictionary\n")
       sys.exit(1)
    sLogs = None
    sDictionary = None
