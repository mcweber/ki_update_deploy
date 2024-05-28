import ai_update_module as ai

def main() -> None:

    while True:
        print("-"*50)
        print(f"Anzahl Mails: {ai.collection_mail_pool.count_documents({})} Anzahl Artikel: {ai.collection_artikel_pool.count_documents({})}")
        print(f"Anzahl Artikel ohne Summary: {ai.collection_artikel_pool.count_documents({'summary': ''})}")
        print(f"Anzahl Artikel ohne Embeddings: {ai.collection_artikel_pool.count_documents({'summary_embeddings': {}})}")
        print("-"*50)
        print("Befehle [M]ailsuche [V]ektorsuche [I]mport [E]xtract [G]enerate abstracts C[r]eate Embeddings [C]lear screen E[x]it")
        befehl = input("Sucheingabe: ")
        print("-"*50)

        if befehl.upper() == "X":
            break

        elif befehl.upper() == "C":
            ai.os.system('cls' if ai.os.name == 'nt' else 'clear')

        elif befehl.upper() == "I":
            input_list = ai.fetch_emails("tldr")
            print("Mail eingelesen.")
            neu_count, doubl_count = ai.add_new_emails(input_list)
            print(f"{neu_count} Mails in Datenbank gespeichert [{doubl_count} Doubletten].")

        elif befehl.upper() == "E":

            cursor, count = ai.text_search_emails("")
            
            for record in cursor:
                
                if record.get("processed") == True:
                    continue

                print(f"[{record.get('date')}] {record.get('title')[:50]}")
                datum, urls = ai.fetch_tldr_urls(record)
                neu_count, doubl_count = ai.add_urls_to_db("tldr", datum, urls)
                ai.collection_mail_pool.update_one({"_id": record.get('_id')}, {"$set": {"processed": True}})
                print(f"{neu_count} URLs in Datenbank gespeichert [{doubl_count} Doubletten].")

        elif befehl.upper() == "G":
            for i in range(10):
                print(f"Generating abstracts {i}/10")
                ai.generate_abstracts(100)
                i += 1

        elif befehl.upper() == "R":
            ai.generate_embeddings(max_iterations=0)

        elif befehl.upper() == "M":
            cursor, count = ai.text_search_emails(input("Mail Search: "))
            ai.print_results(cursor)

        elif befehl.upper() == "V":
            
            result = ai.vector_search_artikel(input("Vector Search: "))
            ai.print_results(result)

        else:

            cursor, count = ai.text_search_artikel(befehl)
            ai.print_results(cursor)


if __name__ == "__main__":
    main()