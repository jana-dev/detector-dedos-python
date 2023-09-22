import cv2
import mediapipe as mp

capture_webcam = cv2.VideoCapture(0) #inicializa a captura de video 0 é padrão/disponível

mp_hands = mp.solutions.hands #módulo que detecta os 21 pontos das mãos
mp_drawing = mp.solutions.drawing_utils #módulo, desenha os pontos detectados da mão
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) #sensibilidade da detecção


tipIds = [4, 8, 12, 16, 20] #id de referencia das pontas dos dedos

def countFingers(frame, hand_landmarks, handNo=0):
    
    if hand_landmarks: #verifica se nao é  nulo
        # Obtenha todos os pontos de referência da mão VISÍVEL
        landmarks = hand_landmarks[handNo].landmark
        #print(landmarks)

        fingers = [] #será usada para rastrear o estado de cada dedo (aberto ou fechado).

        #itera sobre os índices das pontas dos dedos
        for lm_index in tipIds:
                # Obtenha os valores y da ponta e da parte inferior do dedo
                finger_tip_y = landmarks[lm_index].y 
                finger_bottom_y = landmarks[lm_index - 2].y
                #print(finger_tip_y)
                #print(finger_bottom_y)
                
                # obtém as coordenadas X da ponta do dedo e da parte inferior do dedo. Isso é usado para verificar o estado do polegar separadamente.
                thumb_tip_x = landmarks[lm_index].x
                thumb_bottom_x = landmarks[lm_index - 2].x

                # Verifique se ALGUM DEDO está ABERTO ou FECHADO
                if lm_index !=4: #aqui retira o polegar para tratar diferente os dedos
                    if finger_tip_y < finger_bottom_y:
                        fingers.append(1) #adiciona 1 dedo aberto a lista
                        #print("DEDO com id ",lm_index," está Aberto")

                    if finger_tip_y > finger_bottom_y:
                        fingers.append(0) #adiciona 0 para dedo fechado na lista
                        #print("DEDO com id ",lm_index," está Fechado")
                else: #agora verifica o polegar
                    if thumb_tip_x > thumb_bottom_x:
                        fingers.append(1)
                        #print("POLEGAR está Aberto")

                    if thumb_tip_x < thumb_bottom_x:
                        fingers.append(0)
                        #print("POLEGAR está Fechado")


        # print(fingers)
        totalFingers = fingers.count(1) #para contar quantos 1 (dedos abertos) tem na lista

        # Exiba o texto
        text = f'Dedos: {totalFingers}'

        cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)


# Defina uma função para Desenhar as conexões entre os pontos de referência
def drawHandLanmarks(frame, hand_landmarks):

    if hand_landmarks:
      for landmarks in hand_landmarks:    
        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)


while True:
    success, frame = capture_webcam.read() #.read() le um frame da webcam
    frame = cv2.flip(frame, 1) #inverte

    # Execute a detecção de mãos no frame
    results = hands.process(frame)
    # Obtenha a posição do ponto de referência do resultado processado
    hand_landmarks = results.multi_hand_landmarks
    #print(hand_landmarks) ela que nos retorna as coordenadas da mão

    # Desenhe os pontos de referência
    drawHandLanmarks(frame, hand_landmarks)
    # Obtenha a posição dos dedos da mão        
    countFingers(frame, hand_landmarks)
 
    cv2.imshow("Webcam", frame) #exibe o frame em uma janela

    #encerra o loop com espaço
    key = cv2.waitKey(1)
    if key == 32:
        break

#libera a captura de video e fecha todas as janelas abertas
capture_webcam.release()
cv2.destroyAllWindows()