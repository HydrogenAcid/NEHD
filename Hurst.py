# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:54:11 2024

@author: ERICK y BRAULIO

ESPECTRO MULTIFRACTAL
"""
# =============================================================================
# Programa que obtiene el espectro fractal de imagenes de mamografia usando como medidas la media y la varianza 
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.ndimage import binary_dilation
from sympy import isprime
from scipy.stats import kurtosis
from scipy.optimize import curve_fit
plt.rcParams.update({'legend.fontsize': 25})
plt.rcParams.update({'xtick.labelsize': 20, 'ytick.labelsize': 20})

############### obtner falpha #######################
def boxCot (mapaAlfas,histAlphas,Ei,Ef,P): #CORREGIIIIIR
    intervalos=histAlphas[1]
    emes=[]
    prep=[]
    truePrepre=[]
    fil,col=mapaAlfas.shape
    for i in range(intervalos.size-1):
        print(i)
        if i != intervalos.size:
            
            li=intervalos[i]
            ls=intervalos[i+1]
            pmedio=((ls-li)/2)+li
            
            mask1=mapaAlfas>=li
            mask2=mapaAlfas<=ls
            mask=mask1*mask2#mascara binaria
            
            
            
            epsilons=[]
            numCajas=[]
            for e in range(Ei,Ef,P):
                epsilons.append(e)
                n=0
                j=0
                while j<fil:
                    k=0
                    while k<col:
                        if ((j+e)>fil):
                            n+=(np.nansum(np.nansum(mask[j:fil,k:k+e]))>0)
                        elif ((k+e)>col):
                            n+=(np.nansum(np.nansum(mask[j:j+e,k:col]))>0)
                        else:  
                            n+=(np.nansum(np.nansum(mask[j:j+e,k:k+e]))>0)
                        #print(n)
                        k+=e
                    j+=e
                numCajas.append(n)
            # plt.figure()
            # plt.plot(epsilons,numCajas)
            numCajas = np.array(numCajas)
            epsilons = np.array(epsilons)
            
            mask_valid = numCajas > 0
            numCajas = numCajas[mask_valid]
            epsilons = epsilons[mask_valid]
            
            if numCajas.size == 0:
                continue
            
            lnumC = np.log(numCajas)
            leps = np.log(epsilons)
            
            if leps.size == 0 or lnumC.size == 0:
                continue          
            lnumC=np.log(numCajas)
            leps=np.log(epsilons)   
            # plt.figure()
            # plt.plot(leps,lnumC)
            
            m,b=np.polyfit(leps,lnumC,1)
            prep.append(pmedio)
            truePrep=np.mean(mask*mapaAlfas)
            truePrepre.append(truePrep)
            emes.append(m)
            muFit=m*leps+b;
            # if i==80:
            #     plt.figure(0)
            #     plt.imshow(mask,cmap='gray')
            #     plt.axis('off')
            #     plt.show()
                
            #     plt.figure(1)
            #     plt.plot(leps,muFit)
            #     plt.title(r"$log N_\varepsilon (\alpha)  \sim log \varepsilon$ ",fontsize=25,fontweight='bold')
            #     # plt.ylim(0, 1.8) # Limites del eje Y
            #     # plt.xlim(0, 2)
            #    # plt.text(0, 2.5,titulo, fontsize=25, fontweight='bold')
            #     #plt.text(.5, 1.5,'H', fontsize=25, fontweight='bold') 
            #     plt.xlabel(r"$log \varepsilon $",fontsize=25,fontweight='bold')  # Eje X con fracción
            #     plt.ylabel(r"$log N_\varepsilon (\alpha)$",fontsize=25,fontweight='bold')  # Eje Y con fracción
            #     plt.grid(True, linestyle='--', alpha=1)  
            #     plt.scatter(leps,lnumC)
    # plt.figure()
    # plt.scatter(prep,emes)
    # plt.show()
    prep = np.array(prep)
    emes = np.abs( np.array(emes))
    truePrepre=np.array(truePrepre)
    
    # Índices donde numCajas es mayor a 0
    mask_valid = emes >= 0
    # Filtrar ambos arrays por esos índices válidos
    # print(mask_valid)
    # print(emes)
    prep = prep[mask_valid]
    emes = emes[mask_valid]
    truePrepre=truePrepre[mask_valid]
    return (prep,emes,truePrepre) #puntos representativos, M´s,verdaderos puntos representativos

####################        DERIVADAS          ############################
def derivada2DresultadoChistoso(img): ##### YA NO ES UTIL
    X,Y=np.shape(img)
    #img=img-np.mean(img)
    derivada=np.zeros(np.shape(img))
    for i in range(X):
        for j in range(Y):
            if i<j:              
                i1=i    
                k=j-i
                F=img[i][j]*i*j#por el numero total de datos
                diagprin=0
                vec=0
                for i2 in range(i1):
                    diagprin=diagprin+(img[i-i2][j-i2]-img[i-(i2+1)][j-(i2+1)])
                for j2 in range(k):
                    vec=vec+(img[0][k-j2]-img[0][k-(j2+1)])
                derivada[i][j]=F-(diagprin+vec)
            elif i==j:
                 i1=i
                 F=img[i][j]*i*j
                 diagprin=0
                 for i2 in range(i1):                    
                     diagprin=diagprin+(img[i-i2][j-i2]-img[i-(i2+1)][j-(i2+1)])
                 derivada[i][j]=F-(diagprin)
                
            else: 
                i1=j
                k=i-j
                F=img[i][j]*i*j
                diagprin=0
                vec=0
                for i2 in range(i1):
                    diagprin=diagprin+(img[i-i2][j-i2]-img[i-(i2+1)][j-(i2+1)])
                for j2 in range(k):
                    vec=vec+(img[k-j2][0]-img[k-j2-1][0])
                derivada[i][j]=F-(diagprin+vec)
        print(i/X)
    derivada=derivada-np.mean(derivada)
    plt.figure()
    plt.imshow(derivada,cmap='gray')
    plt.colorbar()
    plt.show()
    return derivada
def derivada2D(img): ##### MENOS KAKAKA
    X,Y=np.shape(img)
    #img=img-np.mean(img)
    derivada=np.zeros(np.shape(img))
    for i in range(X):
        for j in range(Y):
            if i<j:              
                i1=i    
                k=j-i
                F=img[i][j]
                diagprin=0
                vec=0
                for i2 in range(i1):
                    diagprin=diagprin+(img[i-i2][j-i2]-img[i-(i2+1)][j-(i2+1)])
                for j2 in range(k):
                    vec=vec+(img[0][k-j2]-img[0][k-(j2+1)])
                derivada[i][j]=F-(diagprin+vec)
            elif i==j:
                 i1=i
                 F=img[i][j]
                 diagprin=0
                 for i2 in range(i1):                    
                     diagprin=diagprin+(img[i-i2][j-i2]-img[i-(i2+1)][j-(i2+1)])
                 derivada[i][j]=F-(diagprin)
                
            else: 
                i1=j
                k=i-j
                F=img[i][j]
                diagprin=0
                vec=0
                for i2 in range(i1):
                    diagprin=diagprin+(img[i-i2][j-i2]-img[i-(i2+1)][j-(i2+1)])
                for j2 in range(k):
                    vec=vec+(img[k-j2][0]-img[k-j2-1][0])
                derivada[i][j]=F-(diagprin+vec)
        print(i/X)
    #derivada=derivada-np.mean(derivada)
    plt.figure()
    plt.imshow(derivada,cmap='gray')
    plt.colorbar()
    plt.show()
    return derivada
def derivadaAnzueto(img): #####DERIVADA ANZUETO
    fil,col=img.shape
    mX=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    mY=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    dx=np.zeros(np.shape(img))
    dy=dx
    for i in range(1,fil-1):
        for j in range(1,col-1):
            w=np.array([[img[i-1][j-1],img[i-1][j],img[i-1][j+1]],[img[i][j-1],img[i][j],img[i][j+1]],[img[i+1][j-1],img[i+1][j],img[i+1][j+1]]])
            dx[i][j]=round(np.sum(w *mX))
            dy[i][j]=round(np.sum(w *mY))
    grad=dx+dy
    plt.figure()
    plt.imshow(grad,cmap='gray')
    plt.colorbar()
    plt.show()
    return (grad) 

############ HURST(INTEGRACION GLOBAL)###########################################
def hurst (img,derivada,Ei,Ef,P): #S global de la derivada
    S=np.std(derivada)
    eles=[]
    fil,col=derivada.shape
    rangesT=[]
    for e in range(Ei,Ef,P):
        eles.append(e/2)
        j=0
        ranges=[]
        while j<fil:
            k=0
            while k<col:
                if ((j+e)>fil):
                    Rloc=np.max(img[j:fil,k:k+e])-np.min(img[j:fil,k:k+e])
                    ranges.append(Rloc/S)
                elif ((k+e)>col):
                    Rloc=np.max(img[j:j+e,k:col])-np.min(img[j:j+e,k:col])
                    ranges.append(Rloc/S)
                else:  
                    Rloc=np.max(img[j:j+e,k:k+e])-np.min(img[j:j+e,k:k+e])
                    ranges.append(Rloc/S)
                k+=e
            j+=e
        
        rangesT.append(np.mean(ranges))
        print(e/Ef)
    lRS=np.log(rangesT)
    leles=np.log(eles)   
    plt.figure()
    plt.plot(leles,lRS)
    hurst,b=np.polyfit(leles,lRS,1)
    return (hurst) 
def hurst2 (img,derivada,Ei,Ef,P):# S local de cada ventana de la derivada
    
    eles=[]
    fil,col=derivada.shape
    rangesT=[]
    for e in range(Ei,Ef,P):
        eles.append(e/2)
        j=0
        ranges=[]
        while j<fil:
            k=0
            while k<col:
                if ((j+e)>fil):
                    Rloc=np.max(img[j:fil,k:k+e])-np.min(img[j:fil,k:k+e])
                    S=np.std(derivada[j:fil,k:k+e])
                    if S==0:
                        S=.0000000000000000000001
                    ranges.append(Rloc/S)
                elif ((k+e)>col):
                    Rloc=np.max(img[j:j+e,k:col])-np.min(img[j:j+e,k:col])
                    S=np.std(derivada[j:j+e,k:col])
                    if S==0:
                        S=.0000000000000000000001
                    ranges.append(Rloc/S)
                else:  
                    Rloc=np.max(img[j:j+e,k:k+e])-np.min(img[j:j+e,k:k+e])
                    S=np.std(derivada[j:j+e,k:k+e])
                    if S==0:
                        S=.0000000000000000000001
                    ranges.append(Rloc/S)
                k+=e
            j+=e
        
        rangesT.append(np.mean(ranges))
        print(e/Ef)
    lRS=np.log(rangesT)
    leles=np.log(eles)   
    plt.figure()
    plt.plot(leles,lRS)
    hurst,b=np.polyfit(leles,lRS,1)
    return (hurst)    

################################### INTEGRACION, ##########################
def integral2 (img):#hace la integral de la ecuacion jajaj, suma la resta de la ventana y la media

    fil,col=img.shape
    integral=np.zeros(np.shape(img))
    # for i in range(fil):
    #     for j in range(col):
    #         integral[i][j]=np.sum(img[0:i, 0:j])
    #   #  print(i/fil)
    integral=np.nancumsum(np.nancumsum((img-np.mean(img)), axis=0), axis=1)
    # plt.figure()
    # plt.imshow(integral,cmap='gray')
    # plt.colorbar()
    # plt.show()
    return integral 

################ HURST INTEGRANDO LOCALMENTE, EL BUENO ES  (hurstintegralLocal4) ####################
#INTEGRAL 2 ES LA BUENA (SUMA DE LA RESTA DE LA VENTANA MENOS LA MEDIA), S LOCAL E INTEGRAL LOCAL
def hurstintegralLocal4(img,Ei,Ef):#obtiene el exponente pero integrando cada ventana, S local integral 2 ESTE ES EL CHIDO
    eles=[]
    fil,col=img.shape
    rangesT=[]
    #S=np.std(img)
    e=Ei
    while e<Ef:
        #eles.append(e/2)
        j=0
        ranges=[]
        while j<fil:
            k=0
            while k<col:
                
                if ((j+e)>fil):
                    if not np.nansum(img[j:fil,k:k+e])==0:
                        regionIntegrada=integral2(img[j:fil,k:k+e]-np.nanmean(img))
                        Rloc=np.nanmax(regionIntegrada)-np.nanmin(regionIntegrada)
                        S=np.nanstd(img[j:fil,k:k+e])
                        if S==0:
                            S=.0000000000000000000001
                        ranges.append(Rloc/S)
                elif ((k+e)>col):
                    if not np.nansum(img[j:j+e,k:col])==0:
                        regionIntegrada=integral2(img[j:j+e,k:col]-np.nanmean(img))
                        Rloc=np.nanmax(regionIntegrada)-np.nanmin(regionIntegrada)
                        S=np.nanstd(img[j:j+e,k:col])
                        if S==0:
                            S=.0000000000000000000001
                        ranges.append(Rloc/S)
                else:  
                    if not np.nansum(img[j:j+e,k:k+e])==0:
                        regionIntegrada=integral2(img[j:j+e,k:k+e]-np.nanmean(img))
                        Rloc=np.nanmax(regionIntegrada)-np.nanmin(regionIntegrada)
                        S=np.nanstd(img[j:j+e,k:k+e])
                        if S==0:
                            S=.0000000000000000000001
                        ranges.append(Rloc/S)
                k+=e
            j+=e
        
        if len(ranges) == 0 or np.isnan(ranges).all():
            continue  # o rangesT.append(np.nan)
        else:
            media_local = np.nanmean(ranges)
            if np.isnan(media_local) or np.isinf(media_local):
                continue
            rangesT.append(media_local)
            eles.append(e / 2)

        e=int(e*np.sqrt(np.sqrt(np.sqrt(np.sqrt(e)))))+1
        #print(e/Ef)
    if len(eles) < 2 or len(rangesT) < 2:
        return np.nan
    
    leles = np.log(eles)
    lRS = np.log(rangesT)
    
    if np.any(np.isnan(leles)) or np.any(np.isnan(lRS)) or np.std(leles) == 0 or np.std(lRS) == 0:
        return np.nan
    
    hurst, b = np.polyfit(leles, lRS, 1)
    return hurst
  
   # print(lRS,leles)
    # plt.figure()
    # titulo='<Rl/Sl> ~ (l/2)^(H) '
    # # plt.ylim(0, 1.8) # Limites del eje Y
    # # plt.xlim(0, 2)
    # plt.text(1, 16,titulo, fontsize=25, fontweight='bold') 
    # plt.xlabel('', fontsize=25, fontweight='bold')  
    # plt.ylabel('<Rl/Sl>', fontsize=25, fontweight='bold')  
    # plt.grid(True, linestyle='--', alpha=1)
    # plt.plot(eles,rangesT)
    
   # if
    # plt.figure()
   #  plt.title(r"$log(<\frac{R_l}{S_l}> ) \sim log (\frac{l}{2})^H)$ ",fontsize=25,fontweight='bold')
   #  # plt.ylim(0, 1.8) # Limites del eje Y
   #  # plt.xlim(0, 2)
   # # plt.text(0, 2.5,titulo, fontsize=25, fontweight='bold')
   #  #plt.text(.5, 1.5,'H', fontsize=25, fontweight='bold') 
   #  plt.xlabel(r"$log (\frac{l}{2})$",fontsize=25,fontweight='bold')  # Eje X con fracción
   #  plt.ylabel(r"$log <\frac{R_l}{S_l}>$",fontsize=25,fontweight='bold')  # Eje Y con fracción
   #  plt.grid(True, linestyle='--', alpha=1)  
   #  plt.scatter(leles,lRS)
    
    hurst,b=np.polyfit(leles,lRS,1)
    recta=hurst*leles+b
    #plt.plot(leles,recta)
    return (hurst)
def hurstintegralLocal4sinInt(img,Ei,Ef):#obtiene el exponente pero integrando cada ventana, S local integral 2
    eles=[]
    fil,col=img.shape
    rangesT=[]
    #S=np.std(img)
    e=Ei
    while e<Ef:
       # print(e)
        eles.append(e/2)
        j=0
        ranges=[]
        while j<fil:
            k=0
            while k<col:
                
                if ((j+e)>fil):
                    if not np.nansum(img[j:fil,k:k+e])==0:
                    #regionIntegrada=integral2(img[j:fil,k:k+e])
                        Rloc=np.nanmax(img[j:fil,k:k+e])-np.nanmin(img[j:fil,k:k+e])
                        S=np.nanstd(img[j:fil,k:k+e])
                        # if S==0:
                        #     S=.0000000000000000000001
                        ranges.append(Rloc/S)
                elif ((k+e)>col):
                    if not np.nansum(img[j:j+e,k:col])==0:
                  #  regionIntegrada=integral2(img[j:j+e,k:col])
                        Rloc=np.nanmax(img[j:j+e,k:col])-np.nanmin(img[j:j+e,k:col])
                        S=np.nanstd(img[j:j+e,k:col])
                        # if S==0:
                        #     S=.0000000000000000000001
                        ranges.append(Rloc/S)
                else: 
                    if not np.nansum(img[j:j+e,k:k+e])==0:
                    #regionIntegrada=integral2(img[j:j+e,k:k+e])
                        Rloc=np.nanmax(img[j:j+e,k:k+e])-np.nanmin(img[j:j+e,k:k+e])
                        S=np.nanstd(img[j:j+e,k:k+e])
                        # if S==0:
                        #     S=.0000000000000000000001
                        ranges.append(Rloc/S)
                k+=e
            j+=e
        
        rangesT.append(np.mean(ranges))
        e=int(e*(np.sqrt(np.sqrt(e))))+1
        
    lRS=np.log(rangesT)
    leles=np.log(eles)   
    
    
    # plt.figure()
    # plt.plot(leles,lRS)
    hurst,b=np.polyfit(leles,lRS,1)
    return (hurst)
def hurstintegralLocal1D(F,Ei,Ef,P):#obtiene el exponente pero integrando cada ventana, S local integral 2
    eles=[]
    N=len(F)
    rangesT=[]
    #S=np.std(img)
    for e in range(Ei,Ef,P):
        eles.append(e/2)
        j=0
        ranges=[]
        while j<N:
                
            if ((j+e)>N):
                regionIntegrada=np.cumsum(F[j:N])
                Rloc=np.max(regionIntegrada)-np.min(regionIntegrada)
                S=np.std(F[j:N])
                if S==0:
                    S=.0000000000000000000001
                ranges.append(Rloc/S)
            
            else:  
                regionIntegrada=np.cumsum(F[j:j+e])
                Rloc=np.max(regionIntegrada)-np.min(regionIntegrada)
                S=np.std(F[j:j+e])
                if S==0:
                    S=.0000000000000000000001
                ranges.append(Rloc/S)
            j+=e
        
        rangesT.append(np.mean(ranges))
        #print(e/Ef)
    lRS=np.log(rangesT)
    leles=np.log(eles)   
    plt.figure()
    plt.plot(leles,lRS)
    hurst,b=np.polyfit(leles,lRS,1)
    return (hurst)

########### RANDOMS ########################################################
def generadordeRandoms(num, fil,col): #### SOLO ES PARA MOSTRAR PROCESO
    num_arrays = num
    shape = (fil,col )
    
    # Crear una lista para almacenar los arrays
    arrays = []
    
    # Generar 30 arrays 2D aleatorios diferentes con distribución normal y semillas diferentes
    for i in range(num_arrays):
        seed = int(time.time() ) + i 
        rng = np.random.RandomState(seed)
        array = rng.randn(*shape)
        # Valores aleatorios con distribución normal
        arrays.append(array)
    return arrays
def hurstpromediado(arreglos,Ei,Ef,P): #### SOLO ES PARA MOSTRAR PROCESO
    plt.close('all')
    hs=[]
    for i in arreglos:
        hs.append(hurstintegralLocal4sinInt(i,Ei,Ef,P)) #### SOLO ES PARA MOSTRAR PROCESO
    return hs,np.mean(hs)       
def hurstporpixel (img, pixel,Tcaja):#### SOLO ES PARA MOSTRAR PROCESO
    localidad=img[pixel[0]:pixel[0]+Tcaja-1,pixel[1]:pixel[1]+Tcaja-1]
    return hurstintegralLocal4(localidad,4,Tcaja,1)

#################### ESTE ES EL METODO PARA LA IMAGEN COMPLETA, PROBAR EL 1 Y EL 2 OBTENCION DE MAPAS###########    
def procesoPorVentana2(img,Tventana): #HURST SIN INTEGRAR 
    #bina=img>=1
    
    X,Y=np.shape(img)
    cvent=int(np.ceil(Tventana/2))
    inicio=[cvent-1,cvent-1]
    final=[X-cvent+1,Y-cvent+1]
    resultado=np.zeros((X,Y))
    for i in range(inicio[0],final[0]):
        
        for j in range(inicio[1],final[1]):             
            cajaCalcular=img[i-(cvent-1):i+(cvent-1),j-(cvent-1):j+(cvent-1)]
            resultado[i,j]=hurstintegralLocal4sinInt(cajaCalcular,2,10)
        print(i/final[0])
    resultado=resultado[inicio[0]:final[0],inicio[1]:final[1]]
    plt.figure()
    plt.imshow(resultado)
    plt.colorbar()
    plt.show()
    return resultado
def procesoPorVentana1(img,Tventana): #original integral este es el bueno
    X,Y=np.shape(img)
    cvent=int(np.ceil(Tventana/2))
    inicio=[cvent-1,cvent-1]
    final=[X-cvent+1,Y-cvent+1]
    resultado=np.zeros((X,Y))+ np.nan
    for i in range(inicio[0],final[0]):
        for j in range(inicio[1],final[1]):
            if not np.isnan(img[i][j]):
                cajaCalcular=img[i-(cvent-1):i+(cvent-1),j-(cvent-1):j+(cvent-1)]
                resultado[i,j]=abs(hurstintegralLocal4(cajaCalcular,2,10))
        #print(i/final[0])
    resultado=resultado[inicio[0]:final[0],inicio[1]:final[1]]
    # plt.figure()
    # plt.imshow(resultado)
    # plt.colorbar()
    # plt.show()
    return resultado

################### PROCESOS POR CARPETA// MOVER //################################
def procesar_carpeta(ruta_carpeta, Tventana):
    archivos_imagenes = [f for f in os.listdir(ruta_carpeta) if f.endswith(('.png'))]
    for archivo in archivos_imagenes:
        ruta_imagen = os.path.join(ruta_carpeta, archivo)
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        resultado = procesoPorVentana1(img, Tventana)
        
        nombre_salida = f'Hurst_{os.path.splitext(archivo)[0]}2.npy'
        ruta_archivo_salida = os.path.join(ruta_carpeta, nombre_salida)
        np.save(ruta_archivo_salida, resultado)       
def cargarCarpeta(ruta_carpeta):
    Archivos=[f for f in os.listdir(ruta_carpeta) if f.endswith(('.npy'))]
    j=0
    for i in Archivos:
        Archivos[j]= np.load(i)
        j+=1
    return Archivos

################### OPERACIONES A LOS MAPAS y vectores de mapas #######################################    
def valoresImportantes(img):  #  retorna los intervalos de clase de la imagen, el falpha
#                                el histograma, y la iagen con hurst de 0 a 2
    fil =np.where(img <= 3, img, np.nan)
    fil =np.where( fil >=0, fil, np.nan) #entre 0 y 2
    
    filhist = img[~np.isnan(img)]
    valHist=np.histogram(filhist,100)
    espectro_Multifractal1 = boxCot(fil, valHist, 5, 25, 5)
    caracteristicas1 = np.abs(espectro_Multifractal1[0])
    fh = np.abs(espectro_Multifractal1[1])
    # delta_alpha, f_max, alpha_0, asimetria, curtosis_f, curvatura=analizar_espectro_multifractal(caracteristicas1, fh)
    return caracteristicas1,fh,valHist,fil
def valoresPorVector(vector): #obtiene los valores omprtantes por cada elemento (mapa) de un vector de estos mismos
    ValoresVector=[]
    for i in range(len(vector)):
        ValoresVector.append(valoresImportantes(vector[i]))
    return ValoresVector
def valoresConNombre(diccionario):
    ValoresVector=[]
    for i in diccionario.keys():
        vectorUnion=[]
        vectorUnion.append(i)
        vectorUnion.append(valoresImportantes(diccionario[i]))
        ValoresVector.append(vectorUnion)
    return ValoresVector    
def histograma(mapa,mostrar): #hace el histograma individual y lo muestra si quieres xd, 100 bins
   # plt.figure()
    plt.ioff()
    mapa = mapa[~np.isnan(mapa)]
    n, bins,_=plt.hist(mapa,100)
    plt.close() 
    if mostrar==1:
        plt.title('Exponente de hurst presente en la imagen')
        plt.xlabel('Hurst')
        plt.ylabel('Frecuencia')
        plt.show()
    return n,bins
def maxminTodas(vector): #obtiene el intervalo de clase minimo y maximo de todos los elementos (mapas)
    maximos=[]
    minimos=[]
    for i in vector:
        maximos.append(max(i[0]))
        minimos.append(min(i[0]))
    return max(maximos),min(minimos)
def histogramaDelimitado(Vector,mapa,mostrar): #histograma individual dentro de los valores de histograma generales
   # plt.figure()
    maxT,minT=maxminTodas(Vector)
    n, bins, patches =plt.hist(mapa,np.linspace(minT, maxT, 100))
    if mostrar==1:
        plt.title('Exponente de hurst presente en la imagen')
        plt.xlabel('Hurst')
        plt.ylabel('Frecuencia')
        plt.show()
    return n,bins  
def delimitarHs(vector): #modifica el espectro multifractal, los intervalos de clase y los histogramas
#                         individuales de acuerdo a los nuevos intervalos de clase generales para cada elemento(mapa)
    for i in range(len(vector)):
        HN=(histogramaDelimitado(vector,vector[i][3],0))
        vector[i][2]=HN
        espectro_Multifractal1 = boxCot(vector[i][3], vector[i][2], 5, 25, 5)
        vector[i][0] = np.abs(espectro_Multifractal1[0])
        vector[i][1] = np.abs(espectro_Multifractal1[1])      
def delimitarHsNombre(vector): #modifica el espectro multifractal, los intervalos de clase y los histogramas
#                         individuales de acuerdo a los nuevos intervalos de clase generales para cada elemento(mapa)
    for i in range(len(vector)):
        vector2 = [sublista[1] for sublista in vector]
        HN=(histogramaDelimitado(vector2,vector[i][1][3],0))
        vector[i][1][2]=HN
        espectro_Multifractal1 = boxCot(vector[i][1][3], vector[i][1][2], 5, 25, 5)
        vector[i][1][0] = np.abs(espectro_Multifractal1[0])
        vector[i][1][1] = np.abs(espectro_Multifractal1[1])
def analizar_espectro_multifractal(alpha, f_alpha):

    # Asegurar orden creciente de alpha
    # orden = np.argsort(alpha)
    # alpha = np.array(alpha)[orden]
    # f_alpha = np.array(f_alpha)[orden]

   
    delta_alpha = np.max(alpha) - np.min(alpha)

    # 2. Valor máximo de f(α)
    delta_falpha = np.max(f_alpha)-np.min(f_alpha)

    # 3. α₀: valor donde f(α) es máximo
    alpha_estrella = alpha[np.argmax(f_alpha)]

    # 4. Asimetría
    alpha_l = alpha[0]
    alpha_r = alpha[-1]
    asimetria = (alpha_r - alpha_estrella) / (alpha_estrella - alpha_l) if (alpha_estrella - alpha_l) != 0 else np.nan

    # 5. Curtosis estadística
    curtosis_f = kurtosis(f_alpha, fisher=False)

    # 6. Ajuste parabólico y curvatura
    def parabola(x, a, b, c):
        return a * (x - b)**2 + c

    try:
        popt, _ = curve_fit(parabola, alpha, f_alpha)
        a_fit, b_fit, c_fit = popt
        curvatura = 2 * a_fit
    except:
        curvatura = np.nan

    return  delta_alpha, delta_falpha, alpha_estrella, asimetria, curtosis_f, curvatura
    
def datos_a_partir_de_imagen_PDI(imagen):
    mapa=procesoPorVentana1(imagen,20)
    alphas,f_alphas,valHist,mapa=valoresImportantes(mapa)
    valores=[alphas,f_alphas,valHist,mapa]
    #graficasindividuales(valores)
    delta_alpha, delta_falpha, alpha_estrella, asimetria, curtosis_f, curvatura=analizar_espectro_multifractal(alphas, f_alphas)
    return delta_alpha, delta_falpha, alpha_estrella, asimetria, curtosis_f, curvatura,alphas,f_alphas,mapa
########################### Gráficos ###########################################
def graficasindividuales(valores): #mapa y espectro correspondiente
    hs=valores[0]
    fh=valores[1]
    #histo=valores[2]
    array=valores[3]
    
    fig, ax = plt.subplots(figsize=(10, 6))  
    ax.plot(hs, fh, '-o', linewidth=2.5, color='red')   
    ax.set_title('Espectro Multifractal', fontsize=28, fontweight='bold') 
    ax.set_xlabel('H', fontsize=20)  
    ax.set_ylabel('f(H)', fontsize=20)  
    ax.grid(True, linestyle='--', alpha=0.6)  
    ax.legend()
    plt.show()
    
    plt.figure()
    plt.imshow(array)
    plt.title('Sin Anomalía', fontsize=15)
    colorbar = plt.colorbar()
    colorbar.set_label('HURST', fontsize=15)
    plt.show()
    
    plt.figure()
    histograma(array,1)
def distribucionDeFuerzaVS(valores,valores1): #espectro versus individual
    hs=valores[0]
    fh=valores[1]
    
    hs1=valores1[0]
    fh1=valores1[1]
    fig, ax = plt.subplots(figsize=(10, 6))  
    ax.plot(hs, fh, '-o', linewidth=2.5, color='red', label='with anomaly')  
    ax.plot(hs1, fh1, '-o', linewidth=2.5, color='darkblue', label='without anomaly')  
    ax.set_title('Distribución de fuerza de singularidad', fontsize=20, fontweight='bold') 
    ax.set_xlabel('H', fontsize=15, fontweight='bold')  
    ax.set_ylabel('f(H)', fontsize=15, fontweight='bold')  
    ax.grid(True, linestyle='--', alpha=0.6)  
    ax.legend()
    plt.show()
def graficaTodas(vector,titulo): #grafica todos los espectros de los elementos del vector en un solo plot
    fig, ax = plt.subplots(figsize=(10, 6))
    p=1
    maximo=0
    minimo=0
    for i in vector:
        #pestana='P_'+str(p)
        ax.plot(i[0], i[1], '-o', linewidth=2.5)#,label=pestana)
        if maximo<max(i[0]):
            maximo=max(i[0])
        if minimo>min(i[0]):
            minimo=min(i[0])
        p+=1
    ax.set_ylim(0, 1.8) # Limites del eje Y
    ax.set_xlim(0, 3)
    ax.set_title(titulo, fontsize=26, fontweight='bold')
    ax.set_xlabel('α', fontsize=25, fontweight='bold')  
    ax.set_ylabel('f(α)', fontsize=25, fontweight='bold')  
    #ax.set_xticks(np.arange(0,maximo+ 0.1, 0.1))
    ax.set_xlim(minimo, maximo)
    ax.grid(True, linestyle='--', alpha=1)  
    ax.legend()
    plt.show()
    return vector
def graficaTodasConNombre(vector,titulo): #grafica todos los espectros de los elementos del vector en un solo plot
    fig, ax = plt.subplots(figsize=(10, 6))
    p=1
    maximo=0
    minimo=0
    valor=.1
    for i in vector:
        valor=round(valor,1)
        #pestana='P_'+str(p)
        ax.plot(i[1][0], i[1][1], '-o', linewidth=2.5,label=i[0])
        if maximo<max(i[1][0]):
            maximo=max(i[1][0])
        if minimo>min(i[1][0]):
            minimo=min(i[1][0])
        p+=.1
        valor+=.1
    ax.set_ylim(0, 1.8) # Limites del eje Y
    ax.set_xlim(0, 3)
    ax.set_title(titulo, fontsize=26, fontweight='bold')
    ax.set_xlabel('α', fontsize=25, fontweight='bold')  
    ax.set_ylabel('f(α)', fontsize=25, fontweight='bold')  
    ax.set_xticks(np.arange(0,maximo+ 0.1, 0.1))
    ax.set_xlim(minimo, maximo)
    ax.grid(True, linestyle='--', alpha=1)  
    ax.legend()
    plt.show()
    return vector        
def graficaPromedio(listaDeVectores): #grafica promedio individual de un solo vector
    n=len(listaDeVectores)
    sumax=np.zeros(99)
    sumay=np.zeros(99)
    for i in range(n):
        sumax=sumax+listaDeVectores[i][0]
        sumay=sumay+listaDeVectores[i][1]
    promx=sumax/n
    promy=sumay/n
    fig, ax = plt.subplots(figsize=(10, 6))  
    ax.plot(promx, promy, '-o', linewidth=2.5, color='red')   
    ax.text('Distribución de fuerza de singularidad', fontsize=20, fontweight='bold') 
    ax.set_xlabel('H', fontsize=15)  
    ax.set_ylabel('f(H)', fontsize=15)  
    ax.grid(True, linestyle='--', alpha=0.6,linewidth=1)  
    ax.legend()
    plt.show()   
def PromedioVs(listaSanos,listaEnfermos): #espectro versus de los promedios
    n=len(listaSanos)
    sumax=np.zeros(99)
    sumay=np.zeros(99)
    errorXS=[]
    errorYS=[]
    for i in range(n):
        sumax=sumax+listaSanos[i][0]
        sumay=sumay+listaSanos[i][1]
    promx=sumax/n
    promy=sumay/n
    
    for i in range(len(listaSanos[0][0])):
        valoresX=[]
        valoresY=[]
        for j in listaSanos:
            valoresX.append(j[0][i])
            valoresY.append(j[1][i])
        errorXS.append(np.std(valoresX))
        errorYS.append(np.std(valoresY))
    nE=len(listaEnfermos)
    sumaxE=np.zeros(99)
    sumayE=np.zeros(99)
    i=0
    errorXE=[]
    errorYE=[]
    for i in range(n):
        sumaxE=sumaxE+listaEnfermos[i][0]
        sumayE=sumayE+listaEnfermos[i][1]
    promxE=sumaxE/nE
    promyE=sumayE/nE
    for i in range(len(listaEnfermos[0][0])):
        valoresX=[]
        valoresY=[]
        for j in listaEnfermos:
            valoresX.append(j[0][i])
            valoresY.append(j[1][i])
        errorXE.append(np.std(valoresX))
        errorYE.append(np.std(valoresY))
    fig, ax = plt.subplots(figsize=(10, 6))  
    ax.plot(promx, promy, '-o', linewidth=2.5, color='darkblue', label='Sin Anomalía')  
    ax.plot(promxE, promyE, '-o', linewidth=2.5, color='red', label='Con Anomalía')  
    for xi, yi,EXSi,EYSi in zip(promx, promy, errorXS,errorYS):
        ax.plot([xi, xi], [yi - EYSi, yi + EYSi], linestyle='--', color='darkblue', linewidth=.8)
    for xi, yi,EXSi,EYSi in zip(promxE, promyE, errorXE,errorYE):
        ax.plot([xi, xi], [yi - EYSi, yi + EYSi], linestyle='--', color='red', linewidth=.8)
    # ax.errorbar(promx, promy,yerr=errorYS,xerr=errorXS, fmt='o', capsize=5, ls='--',elinewidth=1, capthick=1, color='red')  
    # ax.errorbar(promxE, promyE,yerr=errorYE,xerr=errorXE, fmt='o', capsize=5, ls='--',elinewidth=1, capthick=1, color='darkblue') 
    ax.set_title('Espectro multifractal promedio',  fontsize=26, fontweight='bold')
    ax.set_xlabel('α', fontsize=25, fontweight='bold')  
    ax.set_ylabel('f(α)', fontsize=25, fontweight='bold') 
    plt.legend(fontsize=30)
    ax.set_ylim(0, 1.8) # Limites del eje Y
    ax.set_xlim(0, 3)
    ax.grid(True, linestyle='--', alpha=1,linewidth=1)  
    

    ax.legend()
    plt.show()
def compararmapas(data1, data2, cmap='viridis'): #comparar mapas individuales
    """
    Muestra dos mapas (matrices) con un único colorbar compartido,
    ocultando los valores de los ejes.

    Args:
        data1 (ndarray): Primer mapa de datos (matriz).
        data2 (ndarray): Segundo mapa de datos (matriz).
        cmap (str): Mapa de colores a usar. Por defecto, 'viridis'.
        vmin (float): Valor mínimo para el colorbar. Si es None, se calcula automáticamente.
        vmax (float): Valor máximo para el colorbar. Si es None, se calcula automáticamente.
    """
  
    vmin = min(np.nanmin(data1), np.nanmin(data2)) 
    vmax = max(np.nanmax(data1), np.nanmax(data2))
    cmap_instance = plt.get_cmap(cmap).copy()  # Copiar el cmap para evitar cambios globales
    cmap_instance.set_bad(color='black')    # Definir el color para NaN
    
    # Crear la figura y los subplots
    fig = plt.figure(figsize=(12, 6))
    gs = plt.GridSpec(1, 3, width_ratios=[1,1,.05], wspace=0.01)  # Dividir en 3 columnas
    
    #cmap.set_bad(color='black')
    # Primer mapa
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(data1, cmap=cmap_instance, vmin=vmin, vmax=vmax)
    ax1.set_title("Sin anomalía", fontsize=20, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    
    # Segundo mapa
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(data2, cmap=cmap_instance, vmin=vmin, vmax=vmax)
    ax2.set_title("Con anomalía", fontsize=20, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Colorbar
    cbar = fig.colorbar(im1, cax=fig.add_subplot(gs[0, 2]))
    cbar.set_label('Exponente de hurst', fontsize=20, fontweight='bold')
    
    # Mostrar
    plt.tight_layout()
    plt.show()
    print(vmin,vmax)
def mapear_todas(vector_de_mapas, multidimension=0, elemento=0, guardar_vector=0,mostrar=0):
    lista_aislada=[]
    if multidimension==1:   
        for i in vector_de_mapas:
            if mostrar==1:
                plt.figure()
                plt.imshow(i[elemento])
            lista_aislada.append(i[elemento])
    else:
        for i in vector_de_mapas:
            if mostrar==1:
                plt.figure()
                plt.imshow(i)
            lista_aislada.append(i)
    if guardar_vector==1:
        return lista_aislada
    
###################### tapar hoyos (quiza inutil despues) #####################
def rellenar_con_promedio(matriz): #rellenan los nan con los promedios de su vecindad
    # Crear una copia para no modificar la original
    matriz_resultado = matriz.copy()
    
    # Obtener las posiciones donde los valores son NaN (valores en blanco)
    filas, columnas = np.where(np.isnan(matriz_resultado))
    
    # Recorrer las posiciones con NaN
    for fila, columna in zip(filas, columnas):
        # Crear una lista para almacenar los vecinos válidos
        vecinos = []
        
        # Recorrer las posiciones vecinas (arriba, abajo, izquierda, derecha y diagonales)
        for i in range(fila - 1, fila + 2):
            for j in range(columna - 1, columna + 2):
                # Ignorar la posición actual y fuera de los límites
                if 0 <= i < matriz_resultado.shape[0] and 0 <= j < matriz_resultado.shape[1] and (i != fila or j != columna):
                    # Verificar si no es NaN
                    if not np.isnan(matriz_resultado[i, j]):
                        vecinos.append(matriz_resultado[i, j])
        
        # Calcular el promedio de los vecinos válidos
        if vecinos:  # Asegurarse de que hay vecinos válidos
            matriz_resultado[fila, columna] = np.mean(vecinos)
    
    return matriz_resultado
def cuch(mapa): #cuchareo completo, no recuerdo que tan bien sirve
    estructura = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]])
    VB=mapa>0
    bindil= binary_dilation(VB,structure=estructura,iterations=64).astype(int)
    bindil=bindil.astype(float)
    bindil[bindil == 0] = np.nan
    final=bindil*mapa
    plt.imshow(final)
    return final
   
################################### aislar tumor #################################
def aislarTumor(imagen): #qila el tumor xd, pero falla cuando esta pegado a las orillas

    if len(imagen.shape) == 3:
        # Convertir a escala de grises (de 3 a 2 dimensiones)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        # Si ya tiene 2 dimensiones, devolver la imagen tal cual
        imagen = imagen.copy()
    imagen = (imagen* 255).astype(np.uint8)
    laplaciano = cv2.Laplacian(imagen, cv2.CV_64F)
    estructura = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
    bini=imagen>4
    bindil= binary_dilation(bini,structure=estructura,iterations=4).astype(int)
    bindil=bindil.astype(float)
    bindil[bindil == 0] = np.nan
    laplaciano=laplaciano*bindil
    imagen_binarizada = np.where((laplaciano >= -2) & (laplaciano<= 2), 1.0, 0.0)
    imagen_binarizada=imagen_binarizada.astype(np.uint8)
    contornos, _ = cv2.findContours(imagen_binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno_mas_grande = max(contornos, key=cv2.contourArea)
    imagen_mas_grande = np.zeros_like(imagen_binarizada)
    cv2.drawContours(imagen_mas_grande, [contorno_mas_grande], -1, 1, thickness=cv2.FILLED)
    inverso=1-imagen_mas_grande
    imagen_mas_grande=imagen_mas_grande.astype(float)
    inverso=inverso.astype(float)
    imagen_mas_grande[imagen_mas_grande == 0] = np.nan
    inverso[inverso == 0] = np.nan
    tumor=imagen_mas_grande*imagen
    filas_sin_nan = np.where(~np.all(np.isnan(tumor), axis=1))[0]
    columnas_sin_nan = np.where(~np.all(np.isnan(tumor), axis=0))[0]
    
    tumor=np.array(tumor[min(filas_sin_nan):max(filas_sin_nan),min(columnas_sin_nan):max(columnas_sin_nan)])
    tejido=np.array(inverso*imagen*bindil)
    plt.figure()
    plt.imshow(tumor)
    plt.figure()
    plt.imshow(tejido)
    return tumor,tejido
def barajear_matriz(img):
    vector_sin_nan = img[~np.isnan(img)].flatten()
    np.random.shuffle(vector_sin_nan) 
    n=len(vector_sin_nan)
    alto = int(np.sqrt(n))  # Tomamos la raíz cuadrada como base inicial
    print(n,alto)
    i=0
    while( n % alto != 0) and (i<=20):
        alto -= 1  # Reducimos hasta encontrar un divisor
        i+=1
    ancho = n // alto  # Calculamos el ancho
    
    # Ajustamos el tamaño del vector para que coincida exactamente con alto * ancho
    nueva_matriz = vector_sin_nan[:alto * ancho].reshape(alto, ancho)  # Solo tomamos los datos necesarios
    ruido = np.random.randint(0, 256, (alto, ancho), dtype=np.uint8)
    # plt.figure()
    # plt.imshow(img)
    # plt.figure()
    # plt.imshow(nueva_matriz)
    # plt.figure()
    # plt.imshow(ruido)
    print(i,ancho*alto,n,ancho,alto)
    return (img,nueva_matriz,ruido)

def generar_ruido_gaussiano(height=256, width=256, mean=0, std=0.1, scale_min=0, scale_max=1):
    """Genera una imagen con ruido gaussiano y muestra su distribución.
    Permite escalar los valores al rango [scale_min, scale_max]."""
    noise = np.random.normal(mean, std, (height, width))
    
    # Escalar los valores al rango [scale_min, scale_max]
    noise = np.clip((noise - noise.min()) / (noise.max() - noise.min()) * (scale_max - scale_min) + scale_min, scale_min, scale_max)
    
    # Mostrar la imagen del ruido
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(noise, cmap='gray', vmin=scale_min, vmax=scale_max)
    plt.colorbar()
    plt.title("Ruido Gaussiano")
    plt.axis("off")
    
    # Graficar la distribución del ruido
    plt.subplot(1, 2, 2)
    plt.hist(noise.ravel(), bins=50, density=True, color='blue', alpha=0.7)
    plt.xlabel("Valor del píxel")
    plt.ylabel("Densidad")
    plt.title("Distribución del Ruido")
    
    plt.tight_layout()
    plt.show()
    
    return noise
########### COPIA EL SIGUIENTE CODIGO EN LA CONSOLA Y CORRELO PARA CARGAR LOS SANOS,#####
########### ASEGURATE DE QUE TENGAS EL PATH EN LA CARPETA DE LOS SANOS ###################
# ruta_carpeta = r'E:\mamografias\hurst\hurst\resultados\hurstSanosMetodo1'  #PONLE LA RUTA DONDE LOS TENGAS TU
# Sanos=cargarCarpeta(ruta_carpeta)

# ValoresSanosSP=valoresPorVector(Sanos)
# SANOSSP=ValoresSanosSP
# delimitarHs(ValoresSanosSP)

########### COPIA EL SIGUIENTE CODIGO EN LA CONSOLA Y CORRELO PARA CARGAR LOS ENFERMOS,#####
########### ASEGURATE DE QUE TENGAS EL PATH EN LA CARPETA DE LOS ENFERMOS ###################
# ruta_carpetaE = r'E:\mamografias\hurst\hurst\resultados\hurstEnfermosmetodo1' #PONLE LA RUTA DONDE LOS TENGAS TU
# Enfermos=cargarCarpeta(ruta_carpetaE)
# ValoresEnfermosSP=valoresPorVector(Enfermos)
# ENFERMOSSP=ValoresEnfermosSP
# delimitarHs(ValoresEnfermosSP)

######### para graficar################## 
# graficaTodas(ValoresSanosSP,'Sanos')
# graficaTodas(ValoresEnfermosSP,'Enfermos')

# PromedioVs(ValoresSanosSP,ValoresEnfermosSP)