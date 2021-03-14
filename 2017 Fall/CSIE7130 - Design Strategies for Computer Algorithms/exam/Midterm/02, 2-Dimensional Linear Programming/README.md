# Problem 02 2-Dimensional Linear Programming
---
**Description**

在二維平面上，使用 `prune and search` 技巧，實作線性規劃

**Input Format**

第一行包含一個正整數 $n \leq 10^5$，代表限制條件的個數，下一行開始每行包含三個整數 $−300 \leq a,b,c \leq 300$，代表 $ax+by \leq c$ , 其中 $a^2 + b^2 > 0$。

**Output Format**

請輸出滿足所有限制條件的最小 $y$ 值（四捨五入至整數），若沒有解，請輸出 `NA`，若為負無限大，請輸出 `-INF`。 

**Hint**

每筆時限為 `1000ms`，記憶體上限為 `64000KB`。

