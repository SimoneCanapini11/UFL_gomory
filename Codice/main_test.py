import time
from generatore_istanze import genera_istanza_ufl
from ufl_gmic import solve_ufl_with_gomory_gmi
#from ufl_fgc import solve_ufl_with_fgc

def esegui_test():
    print("\nGenerazione del dataset in corso...")
    
    # Generazione Dataset
    dataset = {}
    
    # ISTANZE PICCOLE (n_clienti = 20)
    dataset['P1_Bilanciata']      = genera_istanza_ufl(20, 5,  "bilanciato",           seed=42)
    dataset['P2_SetupDom']        = genera_istanza_ufl(20, 5,  "setup_dominante",      seed=43)
    dataset['P3_TraspDom']        = genera_istanza_ufl(20, 5,  "trasporti_dominante",  seed=44)
    dataset['P4_RatioBassa']      = genera_istanza_ufl(20, 2,  "bilanciato",           seed=52)  # poche facility
    dataset['P5_RatioAlta']       = genera_istanza_ufl(20, 16, "bilanciato",           seed=53)  # tante facility

    # ISTANZE MEDIE (n_clienti = 50-60)
    dataset['M1_Bilanciata']      = genera_istanza_ufl(50, 15, "bilanciato",           seed=45)
    dataset['M2_SetupDom']        = genera_istanza_ufl(50, 15, "setup_dominante",      seed=46)
    dataset['M3_TraspDom']        = genera_istanza_ufl(50, 15, "trasporti_dominante",  seed=47)
    dataset['M4_RatioBassa']      = genera_istanza_ufl(50, 5,  "bilanciato",           seed=48)
    dataset['M5_RatioAlta']       = genera_istanza_ufl(50, 40, "bilanciato",           seed=54)

   # ISTANZE GRANDI (n_clienti = 100)
    dataset['G1_Bilanciata']      = genera_istanza_ufl(100, 30, "bilanciato",          seed=49)
    dataset['G2_SetupDom']        = genera_istanza_ufl(100, 30, "setup_dominante",     seed=50)
    dataset['G3_TraspDom']        = genera_istanza_ufl(100, 30, "trasporti_dominante", seed=51)
    dataset['G4_RatioBassa']      = genera_istanza_ufl(100, 8,  "bilanciato",          seed=55)
    dataset['G5_RatioAlta']       = genera_istanza_ufl(100, 70, "bilanciato",         seed=49)
    
    print(f"Create {len(dataset)} istanze. Avvio risoluzione...\n")
    
    # Lista in cui viene salvato un dizionario per ogni istanza con i risultati
    report_risultati = []

    # Esecuzione algoritmo per ogni istanza
    for nome_istanza, (f, c) in dataset.items():    # .items() restituisce il nome dell'istanza e la tupla (f, c)
        print(f"\n{'='*70}")
        print(f" ESECUZIONE ISTANZA: {nome_istanza}")
        print(f"={'='*70}")
        
        start_time = time.time()
        
        # Chiama la funzione GMI
        risultato = solve_ufl_with_gomory_gmi(f, c, max_cuts=1000, time_limit=300, tol_improvement=0.01)
        #risultato = solve_ufl_with_fgc(f, c, max_cuts=1000, time_limit=300, tol_improvement=0.01)
        
        tempo_totale = time.time() - start_time
        
        # Raccoglie i dati e calcola le statistiche finali
        if risultato is not None:
            model, bound_iniziale, n_tagli = risultato
            bound_finale = model.ObjVal
            
            # Calcolo del miglioramento percentuale totale
            if bound_iniziale > 0:
                miglioramento_totale = ((bound_finale - bound_iniziale) / bound_iniziale) * 100
            else:
                miglioramento_totale = 0.0

            report_risultati.append({
                "Istanza": nome_istanza,
                "LB_Iniziale": bound_iniziale,
                "LB_Finale": bound_finale,
                "Miglioramento": miglioramento_totale,
                "Tagli": n_tagli,
                "Tempo": round(tempo_totale, 2)
            })
        else:
            report_risultati.append({
                "Istanza": nome_istanza,
                "LB_Finale": "Errore"
            })

    # Stampa la tabella finale dei risultati
    print("\n\n" + "*"*95)
    print(f"{'REPORT FINALE DEI TEST UFL - GOMORY MIXED-INTEGER CUTS':^95}")
    # print(f"{'REPORT FINALE DEI TEST UFL - FRACTIONAL GOMORY CUTS':^95}")
    print("*"*95)
    
    # Intestazione della tabella
    print(f"{'Nome Istanza':<22} | {'LB Iniziale':<12} | {'LB Finale':<12} | {'Miglioramento':<15} | {'Tagli':<8} | {'Tempo':<10}")
    print("-" * 95)
    
    # Stampa ogni riga salvata
    for r in report_risultati:
        if r["LB_Finale"] != "Errore":
            miglioramento_str = f"{r['Miglioramento']:.2f}%"
            print(f"{r['Istanza']:<22} | {r['LB_Iniziale']:<12.2f} | {r['LB_Finale']:<12.2f} | {miglioramento_str:<15} | {r['Tagli']:<8} | {r['Tempo']:<8.2f} s")
        else:
            print(f"{r['Istanza']:<22} | {'---':<12} | {'ERRORE':<12} | {'---':<15} | {'---':<8} | {'---':<10}")

if __name__ == "__main__":
    esegui_test()
