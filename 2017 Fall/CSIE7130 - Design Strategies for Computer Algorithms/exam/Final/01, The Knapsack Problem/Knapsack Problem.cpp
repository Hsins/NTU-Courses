#include <cstdio>
#include <queue>
#include <algorithm>
using namespace std;

int capacity;  // Capacity of Knapsack
int itemnum;   // Number of Items

// Define the Stucture for Item which store weight and corresponding value of
// Item
typedef struct Item {
  int value;
  int weight;
} Item;

// Define the Node structure to store information of decision tree
typedef struct Node {
  int level;   // Level of node in decision tree (or index in item[])
  int profit;  // Profit of nodes on path from root to this node (including this
               // node)
  int bound;   // Upper bound of maximum profit in subtree of this node
  int weight;  // The total weight of this Node
} Node;

// Define the Comparison function to sort Item according to value/weight ratio
struct cmp {
  bool operator()(Node& u, Node& v) {
    if (u.bound < v.bound || u.level > v.level) {
      return true;
    } else if (u.bound > v.bound || u.level < v.level) {
      return false;
    }
    return true;
  }
};

// Define the Comparison function to sort Item according to value/weight ratio
bool itemSort(Item a, Item b) {
  double r1 = (double)a.value / a.weight;
  double r2 = (double)b.value / b.weight;
  return r1 > r2;
}

// Declare a priority queue for traversing the node
priority_queue<Node, vector<Node>, cmp> PQ;
Node u, v;

// Define the bound function
int bound(Node u, int n, int capacity, Item item[]) {
  // Items Weights >= Knapsack Capacity
  if (u.weight >= capacity) {
    return 0;
  }
  // Initialize Profits
  int profit_bound = u.profit;
  int j = u.level + 1;
  int total_weight = u.weight;

  while ((j < n) && (total_weight + item[j].weight <= capacity)) {
    total_weight += item[j].weight;
    profit_bound += item[j].value;
    j++;
  }

  if (j < n) {
    profit_bound += (capacity - total_weight) * item[j].value / item[j].weight;
  }

  return profit_bound;
}

int knapsack(int n, int capacity, Item item[]) {
  sort(item, item + itemnum, itemSort);

  // Initialize
  PQ.empty();
  v.level = -1;
  v.profit = v.weight = 0;
  v.bound = bound(v, n, capacity, item);

  // Declare the Lowerbound (the Max Profit)
  int lowerbound = 0;
  PQ.push(v);

  while (!PQ.empty()) {
    // Dequeue a node
    u = PQ.top();
    PQ.pop();

    // If it is starting node, assign level 0
    if (u.level == -1) {
      v.level = 0;
    }
    // If there is nothing on next level
    if (u.level == n - 1) {
      continue;
    }

    // Else if not last node, then increment level, and compute profit of
    // children nodes.
    v.level = u.level + 1;

    // Taking current level's item add current level's weight and value to node
    // u's weight and value
    v.weight = u.weight + item[v.level].weight;
    v.profit = u.profit + item[v.level].value;

    // If cumulated weight is less than W and profit is greater than previous
    // profit, update maxprofit
    if (v.weight <= capacity && v.profit > lowerbound) {
      lowerbound = v.profit;
    }

    // Get the upper bound on profit to decide whether to add v to Q or not.
    v.bound = bound(v, n, capacity, item);

    // If bound value is greater than profit, then only push into queue for
    // further consideration
    if (v.bound > lowerbound) {
      PQ.push(v);
    }

    // Do the same thing, but Without taking the item in knapsack
    v.weight = u.weight;
    v.profit = u.profit;
    v.bound = bound(v, n, capacity, item);
    if (v.bound > lowerbound) PQ.push(v);
  }

  return lowerbound;
}

int main() {
  scanf("%d %d", &capacity, &itemnum);
  Item item[itemnum];

  // Scan the datas (Value and Weight) of each Items
  for (int i = 0; i < itemnum; i++)
    scanf("%d %d", &item[i].value, &item[i].weight);

  printf("%d\n", knapsack(itemnum, capacity, item));

  return 0;
}