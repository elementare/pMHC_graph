# This file contains all pending functions

## Workflow

1 - User pass some arguments to main.py.
### Command Example

Run the script with the following command:
```bash
python3 main.py --mols_path /path/to/pdb_folder --residues_lists /path/to/residues_list.json --centroid_threshold 10 --run_name test_run --association_mode similarity --output_path ./output_directory
```

### Arguments

| Argument                     | Description                                                 | Default             |
|------------------------------|-------------------------------------------------------------|---------------------|
| `--mols_path`                | Path with PDB input files.                                  | `''`                |
| `--centroid_threshold`       | Distance threshold for building the interface graphs.       | `10`                |
| `--run_name`                 | Name for storing results in the output folder.              | `test`              |
| `--association_mode`         | Mode for creating association nodes (identity/similarity).  | `identity`          |
| `--output_path`              | Path to store output results.                              | `'~/'`              |
| `--neighbor_similarity_cutoff` | Threshold for neighbor's similarity.                     | `0.95`              |
| `--rsa_filter`               | Threshold for filtering residues by RSA.                   | `0.1`               |
| `--rsa_similarity_threshold` | Threshold for RSA similarity in association graphs.        | `0.95`              |
| `--residues_lists`           | Path to JSON file containing the pdb residues.             | `None`              |
| `--debug`                    | Activate debug mode.                                       | `False`             |

### Internal Workings


#### Passo a Passo

1. **Inicialização e Argumentos do Usuário**
   - O script começa processando os argumentos de entrada usando a função `parser_args()`. Os argumentos permitem configurar a execução, como os caminhos para os arquivos de entrada, o modo de associação, limiares de distância, entre outros.

2. **Carregamento de Arquivos PDB**
   - O módulo `pdb_io.py` é utilizado para listar e selecionar os arquivos PDB presentes no diretório especificado pelo usuário:
     - `list_pdb_files(pdb_dir)` lista todos os arquivos PDB no diretório.
     - `get_user_selection(pdb_files, pdb_dir)` permite ao usuário selecionar quais arquivos PDB serão usados.

3. **Configuração de Grafo para Proteínas**
   - Inicializa a configuração do grafo de proteínas usando funções do módulo `graphein` e parâmetros definidos pelo usuário. 
   - A função `make_graph_config()` cria a configuração de grafo baseada em limiares de distância e outros critérios.

4. **Construção de Grafos de Proteínas**
   - Os grafos são construídos para cada proteína utilizando a classe `Graph` definida no arquivo `graph.py`.
   - Para cada estrutura PDB, um grafo é criado e é feita a extração de subgrafos relevantes, como resíduos expostos ou interações específicas. Isso tudo é feito utilizando o arquivo **`json`** passado no `--residues_lists`

5. **Construção do Grafo Associado**
   - A classe `AssociatedGraph`, também em `graph.py`, é usada para gerar um grafo que associa as proteínas com base em critérios de identidade e similaridade.
   - A função `_gen_associated_graph()` realiza o produto cartesiano modificado dos grafos para identificar nós associados. Para isso, ela chama a função `association_product()` que está no arquivos `tools.py`
  
6. Dentro da função `association_product()` é chamada a função `filter_reduce_maps()`. Essa função cria uma lista de mapas de contato **filtrados** a partir do argumento **`distance_threshold`**, onde $d(x, y) \leq distance\_threshold := 0$. Além disso, é criado um mapa de resíduos único para que possamos identificar os resíduos presentes na matriz de identidade / similaridade geral.

7. **`neighbors_vec = {i: graph_message_passing(...)`** é responsável por criar dicionário chamado **`neighbors_vec`** onde sua chave representa o grafo de uma dada proteína e o seu valor contém outro dicionário associando um resíduo à sua vizinhança através do ***message_passing*** e os ***fatores de atchley***

8. **`neighbors_similarity = create_neighboor_similarity(...)`** é responsável por criar uma matriz que indica a similaridade da vizinhança entre os resíduos de proteínas diferentes. Temos que para 
$$cossim(a, b) := \begin{cases}
0, \ cossim(a, b) \leq \text{similarity\_cutof}\\
1, \ \text{otherwise}
\end{cases}$$

9. Dependendo do modo de associação, temos a criação de uma matriz utilizando a identidade dos resíduos, ou seja, temos que: $A_{ij} = \delta(a_i, a_j)$. No modo de similaridade, nós temos que $A_{ij} = (1 - \delta(a_i, a_j))\cdot S_{ij}$, onde $S_{ij}$ é uma matriz de similaridade dos vizinhos.

10. 
```py

   block_indices = {} 
   all_possible_nodes = []
   reference_graph = len(graphs[0].nodes())

   for i in range(reference_graph):
        
      block_indices[i] = np.where(associated_nodes_matrix[:, i] > 0)[0]
      block_elements = [[i]]

      for start, end in ranges_graph[1:]:
         elements = [index for index in block_indices[i] if start <= index < end]
            
         if not elements:
            break
            
         block_elements.append([index for index in block_indices[i] if start <= index < end])

      else:
         all_possible_nodes.extend(list(itertools.product(*block_elements)))    

   all_possible_nodes = [node for node in all_possible_nodes if check_multiple_chains(node, residue_maps_unique)]
```

Esse bloco de código tem a função de gerar todas os nós associados possíveis através da permutação dos nós semelhantes/iguais. O `block_indices` armazena os índices semelhantes do resíduo **i**. O `block_elements` armazena os índices de maneira separada para cada proteína [[i], [indicesB], [indicesC] ...]. Após isso, é realizado o produto cartesiano dos índices, que é passado para a variável **`all_possible_nodes`**


11. Filtragem dos nós em que os resíduos vem de cadeias diferentes.

12. É chamada a função `generate_edges()` que irá realizar a combinação entre os nós de associação e criar edges quando eles satisfazerem as condições necessárias. Parte do cálculo foi pré-compilado em C++ utilizando Cython. É utilizado paralelização nos loops para agilizar os cálculos.
   - A similaridade entre os nós é calculada utilizando métodos como `cosine_similarity` e funções de verificação de identidade e similaridade em `tools.py`.
   - Nós e arestas são filtrados com base em critérios específicos, como limiares de distância e cadeia.

13. **Adição de Resíduos de Esfera**
   - A função `add_sphere_residues()` é utilizada para adicionar resíduos esféricos aos modelos, que são salvos em novos arquivos PDB para visualização.

14. **Alinhamento e Comparação Estrutural**
    - O alinhamento das cadeias de proteínas é realizado para comparar as estruturas usando a função `align_structures_by_chain()`.

15. **Processamento Final e Salvamento de Resultados**
    - O grafo final é desenhado e os resultados são salvos no diretório de saída especificado pelo usuário.
    - As informações são registradas em um log para análise e depuração.

## Interações entre os Arquivos

- **main.py**: Script principal que coordena o fluxo de trabalho, chama funções de outros módulos e realiza a lógica de controle.
- **pdb_io.py**: Manipula arquivos PDB e gerencia a seleção de proteínas pelo usuário.
- **graph.py**: Define as classes e métodos para construção e manipulação de grafos de proteínas.
- **tools.py**: Contém funções utilitárias para cálculos de distância, similaridade e manipulação de estruturas de proteínas.

## Notas Importantes

- **Correções em Graphein**: Foram feitas correções no código da biblioteca Graphein para resolver problemas com centroides e compatibilidade com DSSP.
- **Modo de Associação**: O script suporta modos de associação por identidade e similaridade, configuráveis pelo usuário.

Este workflow descreve as etapas gerais e as interações entre os componentes do código para realizar a análise de subgrafos em proteínas. Cada passo é fundamental para assegurar que as proteínas sejam processadas corretamente e que os resultados sejam precisos e informativos.

## Lista de implementação

- [x] Filtragem dos nós que não formaram edges levando em conta o grafo de referência
- [x] Implementação da similaridade do RSA pelo valor absoluto da diferença 
- [ ] Gerar as esferas independentemente se o subgrafo tem peptídeo na associação ou não
- [ ] Realização de testes


### Estratégias alternativas

- Algoritmo de clusteriação para extrair os nós de associação
- Árvore de decisão
- Decoding 

