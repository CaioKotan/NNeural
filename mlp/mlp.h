/*

title: NNeural-MLP

auth: Caio A. Nog
date: 10/14/2026

file: mlp.h

*/
#ifndef MLP_H
#define MLP_H

#include <stddef.h>

typedef struct {

    size_t n_layers; //numero de camadas
    //cada camada pode possuir varios neuronios
    /*
    exemplo:
    size_t layers[] = {2, 3, 1};
    nesse caso teriamos 2 entradas, 3 camadas ocultas e 1 neuronio na saida
    */
    size_t *layer_sizes; //quantidade de neuronios em cada camada.

    float **weights; //pesos
    float **biases; //bias dos neuronios

    float **outputs; //valores calculados pelos neuronios durante as predicoes
} MLP;

MLP *mlp_create(const size_t *layer_sizes, size_t n_layers);

const float *mlp_predict(MLP *mlp, const float *inputs);

void mlp_destroy(MLP *mlp);
int mlp_train(
    MLP *mlp,
    const float *inputs,
    const float *targets,
    float learning_rate
);

#endif
