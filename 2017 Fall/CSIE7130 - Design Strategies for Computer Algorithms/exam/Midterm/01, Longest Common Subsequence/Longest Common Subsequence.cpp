#include <bits/stdc++.h>
#define MAXLEN 105
using namespace std;

// Define the comparator functor using in multiset
struct ltstr {
  bool operator()(const char* s1, const char* s2) const {
    return strcmp(s1, s2) < 0;
  }
};

multiset<const char*, ltstr> ALLLCS;  // Store all possible LCS
int LCSLen, LCSNum, len1, len2;
int LCS[MAXLEN][MAXLEN];              // Dynamics Programming Table
char data[MAXLEN];
char string1[MAXLEN], string2[MAXLEN];

void allLCS(int index1, int index2, int lcs) {
  char k;
  if (lcs == LCSLen) {
    data[lcs] = '\0';
    char* tmp = (char*)malloc(sizeof(char) * (lcs + 1));
    strcpy(tmp, data);
    ALLLCS.insert(tmp);
    return;
  }

  if (index1 == len1 || index2 == len2) {
    return;
  }

  for (int i = index1; i < len1; ++i) {
    k = string1[i];
    for (int j = index2; j < len2; ++j)
      if (k == string2[j] && LCS[i + 1][j + 1] == lcs + 1) {
        data[lcs] = k;
        allLCS(i + 1, j + 1, lcs + 1);
      }
  }
}

void DPMatrix() {
  for (int i = 1; i <= len1; ++i)
    for (int j = 1; j <= len2; ++j)
      if (string1[i - 1] == string2[j - 1]) {
        LCS[i][j] = LCS[i - 1][j - 1] + 1;
      } else {
        LCS[i][j] = max(LCS[i - 1][j], LCS[i][j - 1]);
      }
}

int main() {
  scanf("%s%s", string1, string2);
  len1 = strlen(string1);
  len2 = strlen(string2);

  // Initialize DP matrix
  memset(LCS, 0, sizeof(LCS));

  // Dynamics Programming
  DPMatrix();

  // DP Finish, output the result
  LCSLen = LCS[len1][len2];
  allLCS(0, 0, 0);
  LCSNum = ALLLCS.size();
  printf("%d %d\n", LCSLen, LCSNum);

  // Print all possible LCS
  for (const auto& item : ALLLCS) {
    printf("%s\n", item);
  }

  return 0;
}
