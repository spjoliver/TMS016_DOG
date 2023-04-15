## Skapa en utvecklingsmiljö skild från vscode

0. Följande info utgår ifrån att terminalen ligger i /Docker mappen när kommandona nedan körs.

1. Skapa en bild/image från dockerfilen ("Dockerfile") i denna mappen genom att köra kommandot: ```docker build -f Dockerfile -t bild_namn .```. Nu är bilden skapad och containrar kan skapas med specifica inställningar.

2. För att skapa en container med lämpliga inställningar så kan följande kommando köras: ```docker run -di -p 8888:8888 -v $(cd .. && pwd):/MVE441_2023_DOG --name container_namn bild_namn```. Mappningen av portarna via -p 8888:8888 kan vara mellan vilka portar som helst, bara man håller koll på det. -di säger till containern att köra i detached och interactive mode, dvs att den körs i bakgrunden och hålls levande fritt från den nuvarande processen. -v kommandot står för volume och kopplar i detta fallet ihop hela lokala repot /MVE441_2023_DOG med en mapp med samma namn i containern via $(cd .. && pwd).

3. För att sedan ansluta sig till containern man skapat så kan följande kommando köras: ```docker exec -it container_namn /bin/bash``` där -it gör så att man ansluter till dess terminal. 

4. För att kunna jobba interaktivt med python koden så kan man nu köra kommandot: ```jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root```
    som slår upp en jupyter notebook på den kopplade porten, vilket sen kan accesas från den lokala datorns browser vid localhost:8888.

5. Ctrl-d gör att man hoppar av containern till sin lokala terminal, men containern kör fortfarande i bakgrunden.

