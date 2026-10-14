/*

title: NNeural-Perceptron

auth: Caio A. Nog
date: 10/14/2026

file: perceptron.c

*/
#include "perceptron.h"

#include <stdlib.h>

int perceptron_init(Perceptron *p, size_t n_inputs){ //p e a estrutura que vamos utilizar, se refere a qualquer perceptron que queremos inicializar, usamos ponteiros para modificarmos a estrutura original, nao criarmos uma nova
    size_t i;
    if (p == NULL || n_inputs == 0) //caso p nao seja um objeto valido, retorna 0
        return 0;

    p->n_inputs = 0;// esse "->" serve para apontar para as propriedades da struct "p", por ora define como 0
    p->weights = NULL; //pesos por ora definidos como NULL
    p->bias = 0.0f; //bias por ora definido como 0

    p->weights = (float *)calloc(n_inputs, sizeof(float)); //calloc reserva memoria suficiente para os pesos acompanharem o numero de entradas, caso haja 3 n_inputs, teremos 3 valores float para pesos
    if (p->weights == NULL)
        return 0; //caso a alocacao acima falhe, retorna 0

    p->n_inputs = n_inputs;//apenas define o n_inputs de "p" como o mesmo dos parametros na funcao init

    //assumindo que a alocacao deu certo
    for (i = 0; i < n_inputs; ++i)
        p->weights[i] = 0.0f;//para cada peso, define como 0.0f

    return 1;
}

//aqui e o calculo do perceptron usando a funcao de degrau
//funcao de degrau esta disponivel no arquivo "A.txt"

int perceptron_predict(const Perceptron *p, const float *inputs){
    size_t i;
    float sum;

    if (p == NULL || p->weights == NULL || inputs == NULL)
        return 0;

    sum = p->bias;

    for (i = 0; i < p->n_inputs; ++i)
        sum += p->weights[i] * inputs[i];

    return (sum >= 0.0f) ? 1 : 0; //funcao de degrau
}

//funcao para treino do perceptron
//usa um alvo, valores de entrada, e uma taxa de aprendizado para obter pesos
/*

por exemplo, se quer que o perceptron aprenda a funcao logica AND
uma metodologia seria usar os valores:
    float inputs[] = {1.0f, 1.0f};
    int target = 1;
    float learning_rate = 0.1f;

    o target e o valor que deveria retornar com esses inputs.

*/

int perceptron_train(Perceptron *p, const float *inputs,
                     int target, float learning_rate)
{
    size_t i;
    int prediction;
    float error;

    if (p == NULL || p->weights == NULL || inputs == NULL)
        return 0;

    if ((target != 0 && target != 1) || learning_rate <= 0.0f)
        return 0;

    prediction = perceptron_predict(p, inputs);

    error = (float)(target - prediction);

    for (i = 0; i < p->n_inputs; ++i)
        p->weights[i] += learning_rate * error * inputs[i];

    p->bias += learning_rate * error;

    return 1;
}

void perceptron_free(Perceptron *p)
{
    if (p == NULL) return; //caso p nao seja valido, apenas retorna

    //de resto apenas limpa, as informacoes contidas no perceptron
    free(p->weights);
    p->weights = NULL;
    p->n_inputs = 0;
    p->bias = 0.0f;
}