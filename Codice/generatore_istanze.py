import random

def genera_istanza_ufl(n_clienti, m_facilities, scenario="bilanciato", seed=None):
    """
    Genera un'istanza per il problema UFL.
    
    scenari:
    - "bilanciato": Costi setup e trasporti dello stesso ordine di grandezza.
    - "setup_dominante": Costi setup molto alti, trasporti bassi.
    - "trasporti_dominante": Costi setup bassi, trasporti alti.

    """
    if seed is not None:
        random.seed(seed)
        
    f = []
    c = []
    
    for j in range(m_facilities):
        if scenario == "setup_dominante":   
            # Aprire una facility costa molto di più rispetto ai costi di trasporto.
            # Il solver cercherà di aprirne il meno possibile.
            f.append(n_clienti * random.randint(300, 500))
        elif scenario == "trasporti_dominante":
            # Aprire una facility costa poco.
            # Il solver ne aprirà tante per minimizzare le distanze dei clienti.
            f.append(random.randint(10, 50))
        else: # bilanciato
            # Il costo di apertura è paragonabile a una porzione significativa dei trasporti.
            # Il solver dovrà fare un compromesso.
            f.append(n_clienti * random.randint(50, 150))
            
    for i in range(n_clienti):
        riga_costi = []
        for j in range(m_facilities):
            # Manteniamo i costi di trasporto costanti per capire meglio l'impatto del setup
            # oppure li variamo leggermente. Facciamo un range generico valido per tutti:
            if scenario == "setup_dominante":
                riga_costi.append(random.randint(10, 50))
            elif scenario == "trasporti_dominante":
                riga_costi.append(random.randint(200, 500))
            else: # bilanciato
                riga_costi.append(random.randint(50, 250))
        c.append(riga_costi)
        
    return f, c
