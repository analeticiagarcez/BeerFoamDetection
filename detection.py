import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import lfilter

index = 0
bubbleDens = 0
d = 70
fcount = [] # fcount = Frame count
hcount = [] # hcount = Height count
lcount = [] # lcount = Length count
bcount = [] # bcount = Bubble density count
acount = [] # acount = Average radius count
dcount = []

fig, ax = plt.subplots(2, 2)

ax[0, 0].set_ylabel("Altura total da espuma")
ax[0, 1].set_ylabel("Altura da parte superior")
ax[1, 0].set_ylabel("Densidade da espuma")
ax[1, 1].set_ylabel("Raio medio das bolhas")
ax[0, 0].set_xlabel("Segundos")
ax[0, 1].set_xlabel("Segundos")
ax[1, 0].set_xlabel("Segundos")
ax[1, 1].set_xlabel("Segundos")

# Definicao de onde vem o video, 0 = camera padrao do notebook, CAP_DSHOW = capture direct show = imagens vem direto do input
vid = cv.VideoCapture(0, cv.CAP_DSHOW)

fps = vid.get(cv.CAP_PROP_FPS)
tot_frames = vid.get(cv.CAP_PROP_FRAME_COUNT)

while(vid.isOpened()):
    ret, frame = vid.read()

    if ret == False:
        break

    index = index + 1
    fcount.append(index) # fcount contando a quantidade de frames
    
    # Resize para ajustar imagem na tela do PC
    frame = cv.resize(frame, (960, 540))

    # Transformando o frame em escala de cinza
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Definindo os thresholds para a identificacao do branco (espuma)
    ret, thresh = cv.threshold(gray, 150, 255, 1)
    ret, threshLine = cv.threshold(gray, 250, 255, 1)

    # Definindo os contornos da espuma com base no threshold
    contours, h = cv.findContours(thresh, cv.RETR_TREE,  cv.CHAIN_APPROX_NONE)
    contoursLine, h = cv.findContours(threshLine, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Ordena os contornos para tirar o que abrange a imagem toda
    contours = sorted(contours, key = cv.contourArea, reverse = True)
    contoursLine = sorted(contoursLine, key = cv.contourArea, reverse = True)
    contours.pop(0)
    contoursLine.pop(0)

    # Desenho dos contornos da espuma
    cv.drawContours(frame, contours, 0, (0, 255, 0), 2)

    # Fazendo os contornos do quadrado ao redor da espuma e da linha que fica no topo dela
    c = max(contours, key = cv.contourArea)
    cLine = max(contoursLine, key = cv.contourArea)
    x, y, w, h = cv.boundingRect(c)
    xL, yL, wL, hL = cv.boundingRect(cLine)
    
    hcount.append(h)
    lcount.append(yL)

    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.line(frame, (x, yL), (x+w, yL), (255, 0, 0), 2)
    
    # Calculo da regiao interna da espuma
    partX = int(w/2)
    partY = int(h/2)

    bubbleOverlay = gray[partY+y:2*partY+y, partX+x:2*partX+x]

    #Definicao do kernel e filtragem da regiao
    kernel = np.array([(-1, -1, -1),
                       (-1,  9, -1),
                       (-1, -1, -1)])

    bubbleOverlay = cv.filter2D(bubbleOverlay, -1, kernel)

    #Definindo o threshold para a identificacao do branco (espuma)
    ret, threshBubble = cv.threshold(bubbleOverlay, 200, 255, 1)
    cv.rectangle(frame, (int((partX)/2)+x, int((partY)/2)+y), (int(3*partX/2)+x, int(3*partY/2)+y), (255, 255, 0), 2)

    #Aplicacao dos contornos
    contoursBubble, h = cv.findContours(threshBubble, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contoursBubble = sorted(contoursBubble, key = cv.contourArea, reverse = True)
    contoursBubble.pop(0)

    cv.drawContours(frame, contoursBubble, -1, (255, 255, 0), 1, offset = (int((partX/2)+x), int((partY/2)+y)))

    #Somando as areas de cada bolha e calculando densidade
    for bubble in contoursBubble:
        bubbleDens = bubbleDens + cv.contourArea(bubble)

    bcount.append((bubbleDens/(partX*partY)))

    #Calculando raio medio
    if (len(contoursBubble) != 0):
      acount.append(np.sqrt((bubbleDens/len(contoursBubble))/np.pi))
    else:
        acount.append(0)
        
    bubbleDens = 0

    #Exibicao da imagem
    cv.imshow("Video", frame)

    if cv.waitKey(1) == 27:
        break
    
lcount = [(element * -1 + max(lcount)) for element in lcount]

dur = tot_frames/fps
fcountx=np.linspace(0,dur,int(tot_frames))

d_pix = d/np.average(dcount)
hcount = np.array(hcount)*d_pix
lcount = np.array(lcount)*d_pix

#Filtro
n = 15  # quanto maior o n, mais suave a curva
b = [1.0 / n] * n
a = 1
hcount_filt = lfilter(b, a, hcount)
lcount_filt = lfilter(b, a, lcount)
bcount_filt = lfilter(b, a, bcount)
acount_filt = lfilter(b, a, acount)

#Plotando diagramas
plt.legend()
ax[0, 0].plot(fcountx, hcount_filt, 'r-', label = "Altura total da espuma")
ax[0, 1].plot(fcountx, lcount_filt, 'b-', label = "Altura superior")
ax[1, 0].plot(fcountx, bcount_filt, 'c-', label = "Densidade das bolhas")
ax[1, 1].plot(fcountx, acount_filt, 'm-', label = "Raio médio das bolhas")
plt.show()

cv.destroyAllWindows()
