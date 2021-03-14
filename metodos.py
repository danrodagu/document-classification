import os.path
import re
import nltk
from nltk.corpus import stopwords
import unicodedata
import math
import operator
from datetime import datetime


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
directoryTraining = ROOT_DIR + "/entrenamiento"
directoryCategories = ROOT_DIR + "/entrenamiento/categorias"
directoryTexts = ROOT_DIR + "/entrenamiento/textos"
directoryAdditionalTexts = ROOT_DIR + "/adicionales"

def leer_lineas(path):
    f = open(path)
    lines = f.readlines()
    lines = [x.strip() for x in lines]
    f.close()
    return lines

def leer_lineas_con_saltos(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    return lines

def strip_accents(m):
    return ''.join(c for c in unicodedata.normalize('NFD', m)
                   if unicodedata.category(c) != 'Mn')

def abrir_texto_entrenamiento(docname):
    path = directoryTexts + "/" + docname
    f = open(path, 'r')
    m = f.read().lower()
    s = strip_accents(m)
    f.close()
    return s

def abrir_texto_clasificacion(docname):
    path = directoryAdditionalTexts + "/" + docname
    f = open(path, 'r')
    m = f.read().lower()
    s = strip_accents(m)
    f.close()
    return s

def abrir_archivo_categoria(docname):
    path = directoryCategories + "/" + docname
    f = open(path, 'r')
    m = f.read().lower()
    s = strip_accents(m)
    f.close()
    return s

nDocumentos = os.listdir(directoryTexts).__len__()
allLists = []
textosCategorias = {}

for c in os.listdir(directoryCategories):
    categoria = re.sub('.txt', '', c)
    textosCategorias[categoria] = []

    s = abrir_archivo_categoria(c)
    palabrasClave = []
    for l in s.splitlines():
        palabrasClave.append(l)
    allLists.append(palabrasClave)


for filename in os.listdir(directoryTexts):
    categoria = re.sub('[0-9]?[0-9].txt', '', filename)
    copia = textosCategorias.get(categoria)
    copia.append(filename)
    textosCategorias[categoria] = copia



def clasificar_documento_knn(docname, k):
    horaComienzo = datetime.now()
    path = directoryTraining + "/Entrenamiento kNN.txt"
    lines = leer_lineas(path)
    palabras = lines[1].split("[]")
    palabras[0] = palabras[0].lstrip("[")
    palabras[0] = palabras[0].rstrip("]")
    palabras = palabras[0].split(",") # obtengo las palabras con su orden correspondiente

    listaIDF = lines[4].split("[]")
    listaIDF[0] = listaIDF[0].lstrip("[")
    listaIDF[0] = listaIDF[0].rstrip("]")
    listaIDF = listaIDF[0].split(",")  # obtengo los valores idf con su orden correspondiente


    i = 7
    vectores = {}
    for l in lines: # extraigo los vectores resultantes del entrenamiento y los añado a un diccionario
        if l == lines[i]:
            vector = lines[i].split("()")
            vector[0] = vector[0].lstrip("W = (")
            vector[0] = vector[0].rstrip(")")
            vector = vector[0].split(",")
            doc = lines[i - 1].split("---")
            doc = doc[1].lower()
            vectores[doc] = vector
            i += 2


    m = set(stopwords.words('spanish'))
    stemmer = nltk.stem.PorterStemmer()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    frecuencia = {}  # frencuencias de w
    peso = []  # pesos del documento

    j = 0
    for w in palabras:
        df = 0.0

        s = abrir_texto_clasificacion(docname)
        tokens = tokenizer.tokenize(s)
        stems = [stemmer.stem(t) for t in tokens]  # coge la raíz de las palabras
        filtered = list(filter(lambda v: not v in m, stems))  # filtra eliminando las 'palabras vacías'
        filtered2 = list(filter(lambda v: v == w, filtered))  # filtra por la palabra que estamos analizando
        fr = len(filtered2)  # frecuencia de la palabra en el nuevo documento
        frecuencia[w] = fr
        if fr != 0.0:
            df += 1.0

        p = frecuencia.get(w) * float(listaIDF[j])
        j+=1
        if p == 0.0:
            p = 0
        peso.append(p)

    proximidad = {}

    for v in vectores.keys(): # se calcula la similitud con cada uno de los textos de entrenamiento
        numerador = 0
        raiz1 = 0
        raiz2 = 0
        i = 0
        for p in vectores.get(v):
            numerador = numerador + (float(peso[i]) * float(p))
            raiz1 = raiz1 + (float(peso[i]) * float(peso[i]))
            raiz2 = raiz2 + (float(p) * float(p))
            i+=1
        denominador = math.sqrt(raiz1) * math.sqrt(raiz2)
        if denominador != 0.0:
            proximidad[v] = numerador / denominador
        else:
            proximidad[v] = 0.0

    # print("La similitud con el resto de documentos es: " + str(proximidad) + "\n")

    masCercanos = dict(sorted(proximidad.items(), key=operator.itemgetter(1), reverse=True)[:k]) # se escogen los k textos más próximos al nuevo
    print("Los " + str(k) + " textos más próximos son: " + masCercanos.__str__() + "\n")

    sum = {}
    for categoria in textosCategorias:
        sum[categoria] = 0.0

    for h in masCercanos.keys(): # variante de kNN sumando la similitud de los textos más cercanos de una misma categoría
        categoria = re.sub('[0-9]?[0-9].txt', '', h)
        res = sum.get(categoria) + masCercanos.get(h)
        sum[categoria] = res

    resFinal = dict(sorted(sum.items(), key=operator.itemgetter(1), reverse=True)[:1])
    print("Por el método de la variante del kNN determinamos que la clasificación para este documento es: '" + next(iter(resFinal.keys())) + "'\n")

    horaFinal = datetime.now()
    tiempo = (horaFinal - horaComienzo)
    print("Tiempo de ejecución: " + str(tiempo.total_seconds()) + " segundos")

def clasificar_documento_naive_bayes(docname):
    horaComienzo = datetime.now()
    path = directoryTraining + "/Entrenamiento Naive Bayes.txt"
    lines = leer_lineas(path)
    palabras = lines[1].split("[]")
    palabras[0] = palabras[0].lstrip("[")
    palabras[0] = palabras[0].rstrip("]")
    palabras = palabras[0].split(",")  # obtengo las palabras con su orden correspondiente

    pcs = {}
    ptcs = {}

    for categoria in textosCategorias.keys():
        for l in lines:
            if l == ("---" + categoria.upper() + "---"):
                i = lines.index(l)
                pc = lines[i+1].lstrip("P(" + categoria + ") = ")
                pcs[categoria] = float(pc)

                ptc = lines[i+2].split("()")
                ptc[0] = ptc[0].lstrip("P(t|" + categoria + ") = (")
                ptc[0] = ptc[0].rstrip(")")
                ptc = ptc[0].split(",")
                ptcs[categoria] = ptc
                break

    k = set(stopwords.words('spanish'))
    stemmer = nltk.stem.PorterStemmer()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    s = abrir_texto_clasificacion(docname)
    tokens = tokenizer.tokenize(s)
    stems = [stemmer.stem(t) for t in tokens]  # coge la raíz de palabras
    filtered = list(filter(lambda v: not v in k, stems))  # filtra eliminando las 'palabras vacías'
    filtered = list(filter(lambda v: v in palabras, filtered)) # filtra por las palabras del vocabulario que aparecen

    cnb = {}

    for categoria in textosCategorias.keys():
        ptcs2 = ptcs.get(categoria)
        res = math.log10(pcs.get(categoria))

        for m in filtered:
            j = 0
            for w in palabras:
                if m == w:
                    res = res + math.log10(float(ptcs2[j]))
                j+=1
        cnb[categoria] = res

    cnbFinal = dict(sorted(cnb.items(), key=operator.itemgetter(1), reverse=True)[:1])

    print("La probabilidades del nuevo documento de pertenecer a las distintas categorías son: " + str(cnb) + "\n")

    print("Por el método de Naive Bayes determinamos que la clasificación para este documento es: '" + next(
        iter(cnbFinal.keys())) + "'\n")

    horaFinal = datetime.now()
    tiempo = (horaFinal - horaComienzo)
    print("Tiempo de ejecución: " + str(tiempo.total_seconds()) + " segundos")

def entrenamiento_knn():
    horaComienzo = datetime.now()
    res = list(set(allLists[0]) | set(allLists[1]) | set(allLists[2]) | set(allLists[3]) | set(allLists[4]))
    k = set(stopwords.words('spanish'))
    listaIDF = []
    stemmer = nltk.stem.PorterStemmer()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    path = directoryTraining + "/Entrenamiento kNN.txt"

    f = open(path, 'w+')
    f.write("ORDEN DE PALABRAS PARA LOS VECTORES\n")
    f.write("[")
    i=1
    for r in res:
        if i<res.__len__():
            f.write(r + ",")
        else:
            f.write(r + "]\n\n\n")
        i+=1
    for docname in os.listdir(directoryTexts):
        f.write("---" + docname.upper() + "---\n")
        f.write("W = (\n")
    f.close()

    porcentaje_inicial = 0
    porcentaje = 100 / len(res)

    for w in res:
        df = 0.0
        idf = 0.0
        frecuencia = {} # frecuencia de w por documento
        peso = {} # peso de w por documento

        for docname in os.listdir(directoryTexts):
            s = abrir_texto_entrenamiento(docname)
            tokens = tokenizer.tokenize(s)
            stems = [stemmer.stem(t) for t in tokens]  # coge la raíz de palabras que se repiten
            filtered = list(filter(lambda v: not v in k, stems))  # filtra eliminando las palabras de relleno
            filtered2 = list(filter(lambda v: v==w, filtered)) # filtra por la palabra que estamos analizando
            fr = len(filtered2) # frecuencia en el documento
            frecuencia[docname] = fr
            if fr!= 0.0:
                df+=1

        if df!=0.0:
            idf = math.log10(nDocumentos/df)
        if idf == 0.0:
            idf = 0

        listaIDF.append(idf)

        for docname in os.listdir(directoryTexts):
            p = frecuencia.get(docname)*idf
            if p==0.0:
                p=0
            peso[docname] = p
            lines = leer_lineas(path) # aquí se empieza a sobreescribir el archivo de resultados cada vez que obtenemos los valores para una nueva palabra
            i = lines.index("---" + docname.upper() + "---")
            toAdd = str(lines[i + 1] + str(peso.get(docname))) + ","
            lines.insert(i + 2, toAdd)
            lines.remove(lines[i + 1])
            f = open(path, 'w+')
            for l in lines:
                f.write(l + "\n")
            f.close()

        porcentaje_inicial += porcentaje
        print("Entrenamiento realizado al " + "{0:.2f}".format(porcentaje_inicial) + " %")

    lines = leer_lineas_con_saltos(path)
    f = open(path, 'w+')
    lines.insert(2, "\nFRECUENCIA DOCUMENTAL INVERSA DE CADA PALABRA\n")
    toAdd = "["
    j = 1
    for b in listaIDF:
        if j < listaIDF.__len__():
            toAdd = toAdd + str(b) + ","
        else:
           toAdd = toAdd + str(b) + "]"
        j+=1
    lines.insert(3, toAdd)
    for l in lines:
        l = l.replace(",\n", ")\n")
        f.write(l)
    f.close()

    horaFinal = datetime.now()
    tiempo = (horaFinal - horaComienzo)
    print("\n---> Entrenamiento completado con éxito en: " + str(tiempo.total_seconds()) + " segundos <---")

def entrenamiento_naive_bayes():
    horaComienzo = datetime.now()
    res = list(set(allLists[0]) | set(allLists[1]) | set(allLists[2]) | set(allLists[3]) | set(allLists[4]))
    k = set(stopwords.words('spanish'))
    stemmer = nltk.stem.PorterStemmer()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    path = directoryTraining + "/Entrenamiento Naive Bayes.txt"

    f = open(path, 'w+')
    f.write("ORDEN DE PALABRAS PARA LAS PROBABILIDADES\n")
    f.write("[")
    i = 1
    for r in res:
        if i < res.__len__():
            f.write(r + ",")
        else:
            f.write(r + "]\n\n")
        i += 1

    porcentaje_inicial = 0
    porcentaje = 100 / len(textosCategorias.keys())

    for categoria in textosCategorias.keys():
        pc = textosCategorias.get(categoria).__len__() / nDocumentos # probabilidad de una categoría
        f.write("---" + categoria.upper() + "---\n")
        f.write("P(" + categoria + ") = " + str(pc) + "\n")
        f.write("P(t|" + categoria + ") = (")
        numeradores = {}
        denominador = 0
        concatenacionCategoria = []

        for docname in list(textosCategorias.get(categoria)):
            s = abrir_texto_entrenamiento(docname)
            tokens = tokenizer.tokenize(s)
            stems = [stemmer.stem(t) for t in tokens]  # coge la raíz de palabras
            filtered = list(filter(lambda v: not v in k, stems))  # filtra eliminando las palabras de relleno
            concatenacionCategoria = concatenacionCategoria + filtered # concatena todas las palabras de la categoría

        for w in res:
            filtered2 = list(filter(lambda v: v == w, concatenacionCategoria))  # filtra por la palabra que estamos analizando
            fr = len(filtered2)  # frecuencia en el documento
            numeradores[w] = fr + 1
            denominador = denominador + fr + 1

        j = 1
        for w in res:
            ptc = numeradores.get(w) / denominador
            if j < res.__len__():
                f.write(str(ptc) + ",")
            else:
                f.write(str(ptc) + ")\n\n")
            j+=1

        porcentaje_inicial += porcentaje
        print("Entrenamiento realizado para la categoría '" + categoria + "': entrenamiento realizado al " + "{0:.2f}".format(porcentaje_inicial) + " %")
    f.close()

    horaFinal = datetime.now()
    tiempo = (horaFinal - horaComienzo)
    print("\n---> Entrenamiento completado con éxito en: " + str(tiempo.total_seconds()) + " segundos <---")



if __name__ == "__main__":
    # entrenamiento_knn()
    # entrenamiento_naive_bayes()
    clasificar_documento_knn("adicionaltenis2.txt", 7) # Introducir documento y 'k' deseado
    # clasificar_documento_naive_bayes("adicionaltenis2.txt") # Introducir documento




