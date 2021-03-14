# Problem 02 2-Dimensional Linear Programming
---
**Description**

**Input Format**

每組測資第一行包含一個正整數 $2 \leq n \leq 2 \times 10^5$，代表點的個數，接下來 $n$ 行開始每行包含兩個整數 $|x|,|y| \leq 10^9$，代表該點的座標，保證所有點的座標皆相異。

**Output Format**

第一行請輸出兩個數字分別代表最近點對距離的平方以及個數，下一行開始請輸出所有的配對。每組配對有兩個整數，分別代表兩點輸入時的順序，且第一點為序號較小者。輸出時請依第一點序號遞增排序，若第一點序號相等，則依第二點序號遞增排序。

**Hint**

- 需使用 `Divide and Conquer`，時間複雜度必須為 `O(n\log n)`，否則不予記分或成績折半。
- 注意 `int` 能存取的範圍