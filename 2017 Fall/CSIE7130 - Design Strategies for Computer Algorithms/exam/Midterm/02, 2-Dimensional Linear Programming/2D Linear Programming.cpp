#include <cstdio>
#include <limits>
#include <set>
#include <cmath>

#define ERROR 0.00001
using namespace std;

// Declare a multiset x_set to store each line pair x position.
multiset<double> x_set;

// Declare a number to record the number of Upperline and LowerLine
int num_UpperBound = 0;
int num_LowerBound = 0;

// Initialize the x bound to -INF (Left Bound) and INF (Right Bound)
double x_LB = numeric_limits<int>::min();
double x_RB = numeric_limits<int>::max();

typedef struct Node {
  double max;
  double min;
} Node;

Node lowerSlope;
Node upperSlope;

// A line in homogeneous form:
//   ax + by <= c
//   y = kx + t;
typedef struct line {
  double m;                  // m: Slope of Line, -a/b
  double t;                  // t: Intercept of Line, c/b
  double y_value(double x);  // Give the x value and return y value
  line* prev;
  line* next;
} line;

double line::y_value(double x) { return m * x + t; }

line* lowerLine = new line;
line* upperLine = new line;

// Function to initialize the Line Set
void iniline(line* head, double m, double t) {
  head->m = m;
  head->t = t;
  head->prev = NULL;
  head->next = NULL;
}

// Function to add a line to Line Set
void addline(line** head, double m, double t) {
  line* newLine = new line;
  newLine->m = m;
  newLine->t = t;
  newLine->prev = NULL;
  (*head)->prev = newLine;
  newLine->next = *head;
  *head = newLine;
}

// Function to Delete a line to Line Set
void delline(line** line, char c) {
  if ((*line)->next != NULL) (*line)->next->prev = (*line)->prev;
  if ((*line)->prev != NULL)
    (*line)->prev->next = (*line)->next;
  else if (c == 'L')
    lowerLine = (*line)->next;
  else if (c == 'U')
    upperLine = (*line)->next;
}

// Function to find the intersection point of 2 lines
double intersect(line* line1, line* line2, char c) {
  double det = line2->m - line1->m;
  double x = (line1->t - line2->t) / det;
  double y = (line2->m * line1->t - line1->m * line2->t) / det;

  if (abs(det) > ERROR) {
    if (c == 'x')
      return x;
    else if (c == 'y')
      return y;
  } else if (abs(det) <= ERROR && x == 0 && y == 0)
    return 1;
  else
    return 0;
}

// Function to pair each line
void pairline(line* head, char c) {
  line* curr = head;
  while (curr != NULL && curr->next != NULL) {
    line* tmpCurr = curr;
    if (curr->next->next != NULL)
      curr = curr->next->next;
    else
      curr = NULL;

    double x = intersect(tmpCurr, tmpCurr->next, 'x');
    if (x != 0 && x != 1 && x >= x_LB && x <= x_RB) x_set.insert(x);
    if (c == 'L') {
      if (x >= x_RB) {  // Delete the line with larger m
        if (tmpCurr->m < tmpCurr->next->m) {
          delline(&tmpCurr->next, 'L');
          num_LowerBound--;
        } else if (tmpCurr->m > tmpCurr->next->m) {
          delline(&tmpCurr, 'L');
          num_LowerBound--;
        }
      } else if (x <= x_LB) {  // Delete the line with less m
        if (tmpCurr->m > tmpCurr->next->m) {
          delline(&tmpCurr->next, 'L');
          num_LowerBound--;
        } else if (tmpCurr->m < tmpCurr->next->m) {
          delline(&tmpCurr, 'L');
          num_LowerBound--;
        }
      }
    } else if (c == 'U') {
      if (x >= x_RB) {  // Delete the line with less m
        if (tmpCurr->m > tmpCurr->next->m) {
          delline(&tmpCurr->next, 'U');
          num_UpperBound--;
        } else if (tmpCurr->m < tmpCurr->next->m) {
          delline(&tmpCurr, 'U');
          num_UpperBound--;
        }
      } else if (x <= x_LB) {  // Delete the line with larger m
        if (tmpCurr->m < tmpCurr->next->m) {
          delline(&tmpCurr->next, 'U');
          num_UpperBound--;
        } else if (tmpCurr->m > tmpCurr->next->m) {
          delline(&tmpCurr, 'U');
          num_UpperBound--;
        }
      }
    }
  }
}

void prune(line* head, char c) {
  line* curr = head;
  while (curr != NULL && curr->next != NULL) {
    line* tmpCurr = curr;
    if (curr->next->next != NULL)
      curr = curr->next->next;
    else
      curr = NULL;
    double x = intersect(tmpCurr, tmpCurr->next, 'x');
    if (c == 'L') {
      if (x >= x_RB) {
        if (tmpCurr->m < tmpCurr->next->m) {
          delline(&tmpCurr->next, 'L');
          num_LowerBound--;
        } else if (tmpCurr->m > tmpCurr->next->m) {
          delline(&tmpCurr, 'L');
          num_LowerBound--;
        }
      } else if (x <= x_LB) {  // Delete the line with less m
        if (tmpCurr->m > tmpCurr->next->m) {
          delline(&tmpCurr->next, 'L');
          num_LowerBound--;
        } else if (tmpCurr->m < tmpCurr->next->m) {
          delline(&tmpCurr, 'L');
          num_LowerBound--;
        }
      }
    } else if (c == 'U') {
      if (x >= x_RB) {  // Delete the line with less m
        if (tmpCurr->m > tmpCurr->next->m) {
          delline(&tmpCurr->next, 'U');
          num_UpperBound--;
        } else if (tmpCurr->m < tmpCurr->next->m) {
          delline(&tmpCurr, 'U');
          num_UpperBound--;
        }
      } else if (x <= x_LB) {  // Delete the line with larger m
        if (tmpCurr->m < tmpCurr->next->m) {
          delline(&tmpCurr->next, 'U');
          num_UpperBound--;
        } else if (tmpCurr->m > tmpCurr->next->m) {
          delline(&tmpCurr, 'U');
          num_UpperBound--;
        }
      }
    }
  }
}

double boundPoint(line* head, double xm, double y, char c) {
  Node tmp_node = {numeric_limits<int>::min(), numeric_limits<int>::max()};
  if (c == 'L') {
    line* curr = head;
    double tmp = y;
    while (curr != NULL) {
      if (curr->y_value(xm) > tmp) {
        tmp = curr->y_value(xm);
        tmp_node = {curr->m, curr->m};
      } else if (abs(curr->y_value(xm) - tmp) <= ERROR) {
        tmp_node = {max(curr->m, tmp_node.max), min(curr->m, tmp_node.min)};
      }
      curr = curr->next;
    }
    lowerSlope = tmp_node;
    return tmp;
  } else if (c == 'U') {
    line* curr = head;
    double tmp = y;
    while (curr != NULL) {
      if (curr->y_value(xm) < tmp) {
        tmp = curr->y_value(xm);
        tmp_node = {curr->m, curr->m};
      } else if (abs(curr->y_value(xm) - tmp) <= ERROR) {
        tmp_node = {max(curr->m, tmp_node.max), min(curr->m, tmp_node.min)};
      }
      curr = curr->next;
    }
    upperSlope = tmp_node;
    return tmp;
  }
}

// Main Function
int main() {
  // Declare variables in need
  int num_line;
  int a, b, c;

  // Scan input
  scanf("%d", &num_line);

  // Split All the Constraints into Different Part.
  // upperBound: I+, where coefficient b > 0
  // lowerBound: I-, where coefficient b < 0
  // Change x bound, where coefficient b = 0
  bool isNeg = false, isPos = false;
  for (int i = 0; i < num_line; i++) {
    scanf("%d%d%d", &a, &b, &c);

    if (b < 0) {
      if (!isNeg) {
        iniline(lowerLine, (double)-a / b, (double)c / b);
        num_LowerBound++;
        isNeg = true;
      } else {
        addline(&lowerLine, (double)-a / b, (double)c / b);
        num_LowerBound++;
      }
    } else if (b > 0) {
      if (!isPos) {
        iniline(upperLine, (double)-a / b, (double)c / b);
        num_UpperBound++;
        isPos = true;
      } else {
        addline(&upperLine, (double)-a / b, (double)c / b);
        num_UpperBound++;
      }
    } else {
      if (0 < a && a < x_RB)
        x_RB = (double)c / a;
      else if (a > x_LB)
        x_LB = (double)c / a;
    }
  }

  // Iteration
  while (num_LowerBound + num_UpperBound > 2) {
    if (x_RB < x_LB) {
      printf("NA\n");
      break;
    }
    if (lowerLine == NULL) {
      printf("-INF\n");
      break;
    }
    double lower_y = numeric_limits<int>::min();
    double upper_y = numeric_limits<int>::max();

    pairline(lowerLine, 'L');
    pairline(upperLine, 'U');

    // Find x Median
    auto x_Me = *next(x_set.begin(), x_set.size() / 2);

    // Pair each line and find Bound
    lowerSlope = {0.0, 0.0};
    upperSlope = {0.0, 0.0};
    lower_y = boundPoint(lowerLine, x_Me, lower_y, 'L');
    upper_y = boundPoint(upperLine, x_Me, upper_y, 'U');

    // Discuss different cases
    if (lower_y <= upper_y && lowerSlope.min <= lowerSlope.max &&
        lowerSlope.max < 0) {
      x_LB = x_Me;
    }

    if (lower_y <= upper_y && lowerSlope.max >= lowerSlope.min &&
        lowerSlope.min > 0) {
      x_RB = x_Me;
    }

    if (lower_y <= upper_y && lowerSlope.min <= 0 && 0 <= lowerSlope.max) {
      printf("%d\n", (int)round(lower_y));
      return 0;
    }

    if (lower_y > upper_y && lowerSlope.max < upperSlope.min) {
      x_LB = x_Me;
    }

    if (lower_y > upper_y && lowerSlope.min > upperSlope.max) {
      x_RB = x_Me;
    }

    if (lower_y > upper_y && lowerSlope.max >= upperSlope.min &&
        lowerSlope.min <= upperSlope.max) {
      printf("NA\n");
      return 0;
    }

    prune(lowerLine, 'L');
    prune(upperLine, 'U');
    x_set.clear();
  }

  // 左界右界沒有產生交集
  if (x_RB < x_LB) {
    printf("NA\n");
    return 0;
  }
  // 左界右界有交集，上下界各一條，討論斜率與截距
  else if (x_RB >= x_LB && num_LowerBound == 1 && num_UpperBound == 1) {
    if (lowerLine->m * upperLine->m > 0 &&
        abs(upperLine->m) > abs(lowerLine->m)) {
      printf("%d\n", (int)round(intersect(lowerLine, upperLine, 'y')));
      return 0;
    }
    if (lowerLine->m * upperLine->m > 0 &&
        abs(upperLine->m) < abs(lowerLine->m)) {
      printf("-INF\n");
      return 0;
    }
    if (lowerLine->m * upperLine->m < 0 &&
        abs(upperLine->m) < abs(lowerLine->m)) {
      printf("%d\n", (int)round(intersect(lowerLine, upperLine, 'y')));
      return 0;
    }
    if (lowerLine->m * upperLine->m < 0 &&
        abs(upperLine->m) > abs(lowerLine->m)) {
      printf("-INF\n");
      return 0;
    }
    if (abs(lowerLine->m - upperLine->m) < ERROR &&
        lowerLine->t > upperLine->t) {
      printf("NA\n");
      return 0;
    }
    if (abs(lowerLine->m - upperLine->m) < ERROR &&
        lowerLine->t <= upperLine->t) {
      printf("%d\n", (int)round(min(lowerLine->y_value(x_RB),
                                    lowerLine->y_value(x_LB))));
      return 0;
    }
  }
  // 左界右界有交集，上界兩條，討論斜率與截距
  else if (x_RB >= x_LB && num_LowerBound == 2) {
    if (lowerLine->m * lowerLine->next->m < 0) {
      if (intersect(lowerLine, lowerLine->next, 'x') < x_LB) {
        printf("%d\n", (int)round(max(lowerLine->y_value(x_LB),
                                      lowerLine->next->y_value(x_LB))));
        return 0;
      } else if (intersect(lowerLine, lowerLine->next, 'x') > x_RB) {
        printf("%d\n", (int)round(max(lowerLine->y_value(x_RB),
                                      lowerLine->next->y_value(x_RB))));
        return 0;
      } else if (x_LB < intersect(lowerLine, lowerLine->next, 'x') < x_RB) {
        printf("%d\n", (int)round(lowerLine->y_value(
                           intersect(lowerLine, lowerLine->next, 'x'))));
        return 0;
      } else {
        printf("-INF\n");
        return 0;
      }
    }
    if (lowerLine->m >= 0 && lowerLine->next->m >= 0) {
      if (abs(x_LB - numeric_limits<int>::min()) >= ERROR) {
        printf("%d\n", (int)round(max(lowerLine->y_value(x_LB),
                                      lowerLine->next->y_value(x_LB))));
        return 0;
      } else {
        printf("-INF\n");
        return 0;
      }
    }
    if (lowerLine->m <= 0 && lowerLine->next->m <= 0) {
      if (abs(x_RB - numeric_limits<int>::max()) >= ERROR) {
        printf("%d\n", (int)round(max(lowerLine->y_value(x_RB),
                                      lowerLine->next->y_value(x_RB))));
        return 0;
      } else {
        printf("-INF\n");
        return 0;
      }
    }
  }
  // 左界右界有交集，上界一條
  else if (x_RB >= x_LB && num_LowerBound == 1) {
    printf("%d\n",
           (int)round(min(lowerLine->y_value(x_RB), lowerLine->y_value(x_LB))));
    return 0;
  } else if (x_RB >= x_LB && num_LowerBound == 0) {
    printf("-INF\n");
    return 0;
  }

  return 0;
}
