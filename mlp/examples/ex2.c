/*

title: NNeural-MLP

auth: Caio A. Nog
date: 10/14/2026

file: ex2.c

*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../mlp.h"

int main(void)
{
    size_t layers[] = {2, 3, 1};

    float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    float targets[4][1] = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    srand((unsigned int)time(NULL));

    MLP *net = mlp_create(layers, 3);

    if (net == NULL) {
        printf("Erro ao criar a MLP.\n");
        return 1;
    }

    /* Treinamento */
    size_t epochs = 50000;
    float learning_rate = 0.5f;
    size_t epoch, i;

    for (epoch = 0; epoch < epochs; ++epoch) {
        for (i = 0; i < 4; ++i) {
            if (!mlp_train(
                    net,
                    inputs[i],
                    targets[i],
                    learning_rate)) {

                printf("Erro durante o treinamento.\n");
                mlp_destroy(net);
                return 1;
            }
        }
    }

    /* Teste após o treinamento */
    printf("Resultados do XOR:\n");

    for (i = 0; i < 4; ++i) {
        const float *output =
            mlp_predict(net, inputs[i]);

        printf(
            "%.0f XOR %.0f = %.4f\n",
            inputs[i][0],
            inputs[i][1],
            output[0]
        );
    }

    mlp_destroy(net);
    int a;
    scanf("%d", &a);
    return 0;
}
