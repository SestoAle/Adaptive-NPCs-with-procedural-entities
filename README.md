# Deep Policy Networks for NPC Behaviors that Adapt to Changing Design Parameters in Roguelike Games
### Prerequsites

* The code was tested with **Python v3.6**.

* To install all required packages:
    ```
   cd Adaptive-NPCs-with-procedural-entities
   pip install -r requirements.txt
    ```  
* To download and install DeepCrawl **Potions** task:
    ```
    python download_envs.py
    sudo chmod +x envs/DeepCrawl-Dense-Embedding.x86_64
    sudo chmod +x envs/DeepCrawl-Transformer.x86_64
  
 ## Examples
 * To run traning of an agent with **dense embedding module**, run:
    ```
        python deepcrawl_rl.py -et=dense_embedding
    ```
   
 * To run traning of an agent with **transformer module**, run:
    ```
        python deepcrawl_rl.py -et=transformer
    ```

## Game
The final game with more pre-trained agent will be available soon for Android and iOS devices.