/*

title: NNeural-MLP

auth: Caio A. Nog
date: 10/14/2026

file: mlp.c

*/
#include "mlp.h"

#include <math.h>
#include <stdlib.h>

//sigmoid transforma qualquer valor real em 0 ou 1, observe:
/*

Se x for muito negativo, a saída se aproxima de 0.
Se x for 0, a saída será 0.5.
Se x for muito positivo, a saída se aproxima de 1.

além disso,
A função expf() calcula a exponencial de um float e vem de math.h

*/
static float sigmoid(float x){ //por ser uma static, so e acessada aqui
    return 1.0f / (1.0f + expf(-x));
}


//cria a rede MLP
MLP *mlp_create(const size_t *layer_sizes, size_t n_layers){
    size_t l;
    size_t i;
    MLP *mlp;

    if (layer_sizes == NULL || n_layers < 2)
        return NULL;

    for (l = 0; l < n_layers; ++l) {
        if (layer_sizes[l] == 0)
            return NULL;
    }

    mlp = calloc(1, sizeof(MLP));

    if (mlp == NULL)
        return NULL;

    mlp->n_layers = n_layers;

    mlp->layer_sizes = malloc(n_layers * sizeof(size_t));
    mlp->weights = calloc(n_layers - 1, sizeof(float *));
    mlp->biases = calloc(n_layers - 1, sizeof(float *));
    mlp->outputs = calloc(n_layers, sizeof(float *));

    if (mlp->layer_sizes == NULL ||
        mlp->weights == NULL ||
        mlp->biases == NULL ||
        mlp->outputs == NULL) {
        mlp_destroy(mlp);
        return NULL;
    }

    for (l = 0; l < n_layers; ++l) {
        mlp->layer_sizes[l] = layer_sizes[l];

        mlp->outputs[l] =
            calloc(layer_sizes[l], sizeof(float));

        if (mlp->outputs[l] == NULL) {
            mlp_destroy(mlp);
            return NULL;
        }
    }

    for (l = 0; l < n_layers - 1; ++l) {
        size_t n_weights =
            layer_sizes[l] * layer_sizes[l + 1];

        mlp->weights[l] =
            malloc(n_weights * sizeof(float));

        mlp->biases[l] =
            calloc(layer_sizes[l + 1], sizeof(float));

        if (mlp->weights[l] == NULL ||
            mlp->biases[l] == NULL) {
            mlp_destroy(mlp);
            return NULL;
        }

        for (i = 0; i < n_weights; ++i) {
            mlp->weights[l][i] =
                ((float)rand() / (float)RAND_MAX) - 0.5f;
        }
    }

    return mlp;
}

const float *mlp_predict(MLP *mlp, const float *inputs)
{
    size_t l;
    size_t neuron;
    size_t prev;

    if (mlp == NULL || inputs == NULL)
        return NULL;

    for (neuron = 0; neuron < mlp->layer_sizes[0]; ++neuron)
        mlp->outputs[0][neuron] = inputs[neuron];

    for (l = 0; l < mlp->n_layers - 1; ++l) {
        for (neuron = 0;
             neuron < mlp->layer_sizes[l + 1];
             ++neuron) {

            float sum = mlp->biases[l][neuron];

            for (prev = 0;
                 prev < mlp->layer_sizes[l];
                 ++prev) {

                size_t index =
                    neuron * mlp->layer_sizes[l] + prev;

                sum += mlp->weights[l][index]
                     * mlp->outputs[l][prev];
            }

            mlp->outputs[l + 1][neuron] = sigmoid(sum);
        }
    }

    return mlp->outputs[mlp->n_layers - 1];
}

void mlp_destroy(MLP *mlp)
{
    size_t l;

    if (mlp == NULL)
        return;

    if (mlp->weights != NULL) {
        for (l = 0; l + 1 < mlp->n_layers; ++l)
            free(mlp->weights[l]);
    }

    if (mlp->biases != NULL) {
        for (l = 0; l + 1 < mlp->n_layers; ++l)
            free(mlp->biases[l]);
    }

    if (mlp->outputs != NULL) {
        for (l = 0; l < mlp->n_layers; ++l)
            free(mlp->outputs[l]);
    }

    free(mlp->weights);
    free(mlp->biases);
    free(mlp->outputs);
    free(mlp->layer_sizes);
    free(mlp);
}

int mlp_train(
    MLP *mlp,
    const float *inputs,
    const float *targets,
    float learning_rate
)
{
    size_t l, neuron, next;
    size_t last;
    float **deltas;

    if (mlp == NULL || inputs == NULL || targets == NULL)
        return 0;

    if (learning_rate <= 0.0f)
        return 0;

    /* Calcula a saída atual da rede. */
    if (mlp_predict(mlp, inputs) == NULL)
        return 0;

    last = mlp->n_layers - 1;

    /* Reserva espaço para os erros de cada camada. */
    deltas = calloc(mlp->n_layers, sizeof(float *));

    if (deltas == NULL)
        return 0;

    for (l = 1; l < mlp->n_layers; ++l) {
        deltas[l] = calloc(
            mlp->layer_sizes[l],
            sizeof(float)
        );

        if (deltas[l] == NULL) {
            for (size_t k = 0; k < mlp->n_layers; ++k)
                free(deltas[k]);

            free(deltas);
            return 0;
        }
    }

    /* 1. Calcula o delta da camada de saída. */
    for (neuron = 0;
         neuron < mlp->layer_sizes[last];
         ++neuron) {

        float output = mlp->outputs[last][neuron];

        deltas[last][neuron] =
            (targets[neuron] - output)
            * output * (1.0f - output);
    }

    /*
     * 2. Propaga o erro para as camadas ocultas.
     * Os pesos ainda não foram atualizados aqui.
     */
    for (l = last; l > 1; --l) {
        size_t current_layer = l - 1;

        for (neuron = 0;
             neuron < mlp->layer_sizes[current_layer];
             ++neuron) {

            float sum = 0.0f;

            for (next = 0;
                 next < mlp->layer_sizes[l];
                 ++next) {

                size_t index =
                    next * mlp->layer_sizes[current_layer]
                    + neuron;

                sum += deltas[l][next]
                     * mlp->weights[current_layer][index];
            }

            {
                float output =
                    mlp->outputs[current_layer][neuron];

                deltas[current_layer][neuron] =
                    output * (1.0f - output) * sum;
            }
        }
    }

    /* 3. Atualiza os pesos e os vieses. */
    for (l = 0; l < last; ++l) {
        for (neuron = 0;
             neuron < mlp->layer_sizes[l + 1];
             ++neuron) {

            for (next = 0;
                 next < mlp->layer_sizes[l];
                 ++next) {

                size_t index =
                    neuron * mlp->layer_sizes[l] + next;

                mlp->weights[l][index] +=
                    learning_rate
                    * deltas[l + 1][neuron]
                    * mlp->outputs[l][next];
            }

            mlp->biases[l][neuron] +=
                learning_rate * deltas[l + 1][neuron];
        }
    }

    for (l = 0; l < mlp->n_layers; ++l)
        free(deltas[l]);

    free(deltas);

    return 1;
}
