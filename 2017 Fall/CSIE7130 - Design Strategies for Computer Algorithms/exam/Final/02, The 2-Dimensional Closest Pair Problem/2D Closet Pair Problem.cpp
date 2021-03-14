#include <bits/stdc++.h>
#define MAXN 200000
#define ERROR 0.00001

// Declare the Variables
int ansamt = 0;                  // The Amount of Answer Pair
long long distance = LLONG_MAX;  // The Answer Distance (Initial: LLONG_MAX)

// Define a structure to represent Points in 2D Plane.
typedef struct Point {
  long long x;  // x coordinate
  long long y;  // y coordinate
  int index;    // The Number of Points
} Point;
Point point[MAXN];

// Define the structure to store index of answer points
typedef struct AnsIdx {
  int idx1;
  int idx2;
} AnsIdx;
AnsIdx pntpair[MAXN];

// Define the function to calculate the distance between two points
long long dist(Point p1, Point p2) {
  return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

// Define the function to compare the x coordinate
int cmpx(const void* s1, const void* s2) {
  return static_cast<int>(((Point*)s1)->x - (((Point*)s2)->x));
}

// Define the function to compare the y coordinate
int cmpy(const void* s1, const void* s2) {
  return static_cast<int>(((Point*)s1)->y - (((Point*)s2)->y));
}

// Define the function to compare the index
int cmpidx(const void* s1, const void* s2) {
  auto *a = (AnsIdx *)s1, *b = (AnsIdx *)s2;
  if (a->idx1 == b->idx1) {
    return (a->idx2 - b->idx2);
  }
  return (a->idx1 - b->idx1);
}

void ClosestPair(int start, int end) {
  if (end <= start) {
    return;
  }
  if (end - start == 1) {
    long long newdist = dist(point[start], point[end]);
    if (distance - newdist > ERROR) {
      distance = newdist;
      pntpair[0].idx1 = point[start].index;
      pntpair[0].idx2 = point[end].index;
      ansamt = 1;
    } else if (llabs(distance - newdist) < ERROR) {
      pntpair[ansamt].idx1 = point[start].index;
      pntpair[ansamt].idx2 = point[end].index;
      ansamt++;
    }
    qsort(&point[start], static_cast<size_t>(end - start + 1), sizeof(Point),
          cmpy);
    return;
  }

  // Divide the pair into two part
  int idxmid = (start + end) / 2;
  long long xmid = point[idxmid].x;

  // Find the closet pair in each part
  ClosestPair(start, idxmid);
  ClosestPair(idxmid + 1, end);

  int l = 0;
  int r = 0;
  static Point left[MAXN / 2];
  static Point right[MAXN / 2];

  // Left Part
  for (int i = start; i <= idxmid; i++) {
    if ((xmid - point[i].x) <= distance) {
      left[l] = point[i];
      l++;
    }
  }

  // Right Part
  for (int i = idxmid + 1; i <= end; i++) {
    if ((point[i].x - xmid) <= distance) {
      right[r] = point[i];
      r++;
    }
  }

  // Merge
  int from = 0;
  for (int i = 0; i < l; i++) {
    int upcount = 0;
    from = (from - 5) > 0 ? from - 5 : 0;
    for (int j = from; j < r; j++) {
      long long newdist = dist(left[i], right[j]);
      if (distance - newdist > ERROR) {
        distance = newdist;
        pntpair[0].idx1 = left[i].index;
        pntpair[0].idx2 = right[j].index;
        ansamt = 1;
      } else if (llabs(newdist - distance) < ERROR) {
        pntpair[ansamt].idx1 = left[i].index;
        pntpair[ansamt].idx2 = right[j].index;
        ansamt++;
      }
      if (right[j].y > left[i].y) {
        upcount++;
      }
      if (upcount > 3) {
        from = j;
        break;
      }
    }
  }

  // Sort the Point with y coordinate
  qsort(&point[start], static_cast<size_t>(end - start + 1), sizeof(Point),
        cmpy);
}

int main() {
  // Declare the variable to store the amount of points.
  int n;

  // Scan the points data
  scanf("%d", &n);
  for (int i = 0; i < n; i++) {
    scanf("%lld%lld", &point[i].x, &point[i].y);
    point[i].index = i + 1;
  }

  // Sort the points by the x coordinate
  qsort(point, static_cast<size_t>(n), sizeof(Point), cmpx);

  // Run the function to find the distance and pairs
  ClosestPair(0, n - 1);

  // Print the distance and amounts of pairs
  printf("%lld %i\n", distance, ansamt);

  for (int i = 0; i < ansamt; i++) {
    if (pntpair[i].idx1 > pntpair[i].idx2) {
      int temp = pntpair[i].idx1;
      pntpair[i].idx1 = pntpair[i].idx2;
      pntpair[i].idx2 = temp;
    }
  }

  // Print all the answer points
  qsort(pntpair, static_cast<size_t>(ansamt), sizeof(AnsIdx), cmpidx);
  for (int i = 0; i < ansamt; i++) {
    printf("%d %d\n", pntpair[i].idx1, pntpair[i].idx2);
  }
  return 0;
}
