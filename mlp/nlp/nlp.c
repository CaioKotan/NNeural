/*
title: NNeural-MLP-NLP
auth: Caio A. Nog
date: 10/14/2026
file: nlp.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../mlp.h"

#define VOCAB_SIZE 250
#define HIDDEN_SIZE 32
#define CONTEXT_SIZE 2
#define MAX_WORDS 64
#define EPOCHS 10000

const char *termos[250] = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
    "e", "com", "nao", "uma", "os", "no", "se", "na", "por", "mais",
    "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "a",
    "seu", "sua", "ou", "ser", "quando", "muito", "ha", "nos", "ja", "esta",
    "eu", "tambem", "so", "pelo", "pela", "ate", "isso", "ela", "entre", "depois",
    "sem", "mesmo", "aos", "ter", "quem", "nas", "me", "esse", "essa", "foram",
    "era", "estao", "bem", "ainda", "onde", "entao", "sobre", "aqui", "outro", "todas",
    "porque", "pode", "dizer", "fazer", "vai", "estar", "minha", "meu", "assim", "cada",
    "qual", "quais", "tudo", "nada", "coisa", "pessoas", "vida", "tempo", "dia", "vez",
    "anos", "mundo", "casa", "trabalho", "forma", "parte", "caso", "lugar", "momento", "maneira",
    "governo", "estado", "pais", "homem", "mulher", "crianca", "familia", "pessoa", "gente", "grupo",
    "sistema", "empresa", "programa", "informacao", "problema", "questao", "direito", "poder", "razao", "fato",
    "exemplo", "tipo", "numero", "ponto", "nivel", "area", "lado", "fim", "inicio", "comeco",
    "meio", "processo", "resultado", "desenvolvimento", "situacao", "condicao", "relacao", "historia", "palavra", "nome",
    "ideia", "pensamento", "conhecimento", "educacao", "escola", "aluno", "professor", "livro", "texto", "lingua",
    "linguagem", "comunicacao", "pergunta", "resposta", "verdade", "realidade", "sentido", "significado", "atencao", "interesse",
    "necessidade", "importancia", "possibilidade", "oportunidade", "experiencia", "atividade", "servico", "produto", "dinheiro", "preco",
    "mercado", "negocio", "compra", "venda", "recurso", "valor", "custo", "conta", "banco", "emprego",
    "salario", "cliente", "usuario", "publico", "sociedade", "comunidade", "populacao", "cultura", "politica", "economia",
    "saude", "seguranca", "tecnologia", "internet", "computador", "celular", "telefone", "rede", "dados", "arquivo",
    "codigo", "software", "maquina", "dispositivo", "energia", "agua", "ar", "terra", "fogo", "luz",
    "som", "imagem", "video", "musica", "arte", "ciencia", "natureza", "ambiente", "animal", "planta",
    "alimento", "comida", "bebida", "corpo", "cabeca", "mao", "olho", "rosto", "coracao", "mente",
    "voz", "pe", "braco", "perna", "noite", "manha", "tarde", "semana", "mes", "ano",
    "hora", "minuto", "segundo", "hoje", "ontem", "amanha", "agora", "antes", "sempre", "nunca"
};
int find_term(const char *word)
{
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (strcmp(termos[i], word) == 0) {
            return i;
        }
    }

    return -1;
}

/* Converte uma frase em indices de palavras conhecidas. */
size_t tokenize(
    const char *text,
    int *indices,
    size_t max_words
)
{
    char buffer[512];

    snprintf(buffer, sizeof(buffer), "%s", text);

    const char *delimiters = " \t\n\r.,!?;:\"()[]{}";
    char *word = strtok(buffer, delimiters);

    size_t count = 0;

    while (word != NULL && count < max_words) {
        int index = find_term(word);

        if (index >= 0) {
            indices[count++] = index;
        }

        word = strtok(NULL, delimiters);
    }

    return count;
}

/* Cria uma entrada Bag of Words com duas palavras de contexto. */
void make_input(
    const int *context,
    float *input
)
{
    for (int i = 0; i < VOCAB_SIZE; i++) {
        input[i] = 0.0f;
    }

    for (int i = 0; i < CONTEXT_SIZE; i++) {
        if (context[i] >= 0) {
            input[context[i]] = 1.0f;
        }
    }
}

/* Retorna o indice da maior saida da rede. */
int argmax(const float *values, size_t count)
{
    int best = 0;

    for (size_t i = 1; i < count; i++) {
        if (values[i] > values[best]) {
            best = (int)i;
        }
    }

    return best;
}

int main(void)
{
    srand((unsigned int)time(NULL));

    /*
     * Exemplos de treinamento.
     * A rede aprende a prever a palavra seguinte.
     *
     * IMPORTANTE:
     * Todas as palavras devem existir em termos[].
     * O tokenizer diferencia maiusculas de minusculas.
     */
    const char *corpus[] = {
        "hoje eu trabalho com computador",
        "eu trabalho com computador",
        "hoje eu trabalho com computador",
        "eu trabalho com computador"
    };

    size_t corpus_size =
        sizeof(corpus) / sizeof(corpus[0]);

    size_t layers[] = {
        VOCAB_SIZE,
        HIDDEN_SIZE,
        VOCAB_SIZE
    };

    MLP *net = mlp_create(layers, 3);

    if (net == NULL) {
        fprintf(stderr, "Erro ao criar a MLP.\n");
        return 1;
    }

    float input[VOCAB_SIZE];
    float target[VOCAB_SIZE];

    int words[MAX_WORDS];

    float learning_rate = 0.1f;

    /* Treinamento */
    for (size_t epoch = 0; epoch < EPOCHS; epoch++) {
        for (size_t s = 0; s < corpus_size; s++) {

            size_t count = tokenize(
                corpus[s],
                words,
                MAX_WORDS
            );

            /*
             * Usa as duas palavras anteriores
             * para prever a proxima palavra.
             */
            for (size_t i = CONTEXT_SIZE; i < count; i++) {

                int context[CONTEXT_SIZE] = {
                    words[i - 2],
                    words[i - 1]
                };

                make_input(context, input);

                /* Target one-hot. */
                for (int j = 0; j < VOCAB_SIZE; j++) {
                    target[j] = 0.0f;
                }

                target[words[i]] = 1.0f;

                if (!mlp_train(
                        net,
                        input,
                        target,
                        learning_rate)) {

                    fprintf(
                        stderr,
                        "Erro durante o treinamento.\n"
                    );

                    mlp_destroy(net);
                    return 1;
                }
            }
        }
    }

    printf("Treinamento concluido.\n");
    printf("Digite duas palavras para prever a proxima.\n");
    printf("Digite 'sair' para encerrar.\n\n");

    /* Inferencia */
    char sentence[512];

    while (1) {
        printf("> ");

        if (fgets(sentence, sizeof(sentence), stdin) == NULL) {
            break;
        }

        sentence[strcspn(sentence, "\r\n")] = '\0';

        if (strcmp(sentence, "sair") == 0) {
            break;
        }

        size_t count = tokenize(
            sentence,
            words,
            MAX_WORDS
        );

        if (count < CONTEXT_SIZE) {
            printf("Informe pelo menos duas palavras conhecidas.\n");
            continue;
        }

        int context[CONTEXT_SIZE] = {
            words[count - 2],
            words[count - 1]
        };

        make_input(context, input);

        const float *output = mlp_predict(net, input);

        if (output == NULL) {
            printf("Erro ao executar a previsao.\n");
            continue;
        }

        int prediction = argmax(output, VOCAB_SIZE);

        printf(
            "Proxima palavra prevista: %s\n",
            termos[prediction]
        );
    }

    mlp_destroy(net);

    return 0;
}