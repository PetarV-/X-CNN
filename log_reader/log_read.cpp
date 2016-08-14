#include <stdio.h>
#include <assert.h>
#include <vector>

using namespace std;

char token[1001];

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    printf("Usage: ./log_read <in_file> <out_file>\n");
    return -1;
  }

  char *in_file = argv[1];
  char *out_file = argv[2];

  FILE *f = fopen(in_file, "r");
  FILE *g = fopen(out_file, "w");

  if (f == NULL || g == NULL)
  {
    printf("Failed to open the files!\n");
    return -2;
  }

  vector<double> train_loss;
  vector<double> train_acc;
  vector<double> test_loss;
  vector<double> test_acc;

  while (fscanf(f, "%s", token) == 1)
  {
    if (strcmp(token, "loss:") == 0)
    {
      double t_l;
      fscanf(f, "%lf", &t_l);
      train_loss.push_back(t_l);
    }
    else if (strcmp(token, "acc:") == 0)
    {
      double t_a;
      fscanf(f, "%lf", &t_a);
      train_acc.push_back(t_a);
    }
    else if (strcmp(token, "val_loss:") == 0)
    {
      double v_l;
      fscanf(f, "%lf", &v_l);
      test_loss.push_back(v_l);
    }
    else if (strcmp(token, "val_acc:") == 0)
    {
      double v_a;
      fscanf(f, "%lf", &v_a);
      test_acc.push_back(v_a);
    }
  }

  assert(test_loss.size() == train_loss.size());
  assert(test_acc.size() == train_acc.size());
  assert(test_loss.size() == test_acc.size());

  fprintf(g, "%lu\n", test_loss.size());

  for (double x : train_loss)
  {
    fprintf(g, "%.4lf ", x);
  }
  fprintf(g, "\n");
  for (double x : train_acc)
  {
    fprintf(g, "%.4lf ", x);
  }
  fprintf(g, "\n");
  for (double x : test_loss)
  {
    fprintf(g, "%.4lf ", x);
  }
  fprintf(g, "\n");
  for (double x : test_acc)
  {
    fprintf(g, "%.4lf ", x);
  }
  fprintf(g, "\n");

  fclose(f);
  fclose(g);

  return 0;
}
