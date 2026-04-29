# Proponowane zadania po przeglądzie kodu

## 1) Zadanie: poprawa literówki
**Obszar:** `tests/test_muse_connection_resilience.py`.

W komentarzu TODO występują literówki i brak polskich znaków (np. `Rozszerzyc`, `ktorym`).
To utrudnia czytelność i obniża jakość dokumentacji technicznej testu.

**Propozycja:**
- poprawić zapis komentarza na poprawną polszczyznę,
- zachować sens TODO, ale doprecyzować kryterium ukończenia.

---

## 2) Zadanie: usunięcie błędu logicznego
**Obszar:** `src/settings.py` (`AppSettings.from_dict`).

Pole `debug_logging_enabled` korzysta z fallbacku do legacy `debug_eeg_file`, ale obecna kolejność może w praktyce nadpisać intencję użytkownika przy niejednoznacznych danych wejściowych (migracja starych i nowych ustawień w jednym pliku).

**Propozycja:**
- doprecyzować regułę migracji i priorytet pól (`debug_logging_enabled` > `debug_eeg_file`),
- dodać jawny etap migracji legacy zamiast łączenia logiki bezpośrednio w `dict.get(...)`,
- dodać walidację typu wejściowego przed migracją.

---

## 3) Zadanie: korekta komentarza/dokumentacji
**Obszar:** `src/game_controller.py` (docstring klasy i sekcja o hysteresis).

Docstring metody `update()` mówi o stałej `HYSTERESIS_COUNT`, ale implementacja używa konfigurowalnego `self.hysteresis_count` (np. ustawianego przez `settings`). To rozbieżność pomiędzy komentarzem a realnym działaniem.

**Propozycja:**
- zmienić opis na „`self.hysteresis_count` kolejnych odczytów”,
- dopisać, że wartość może być nadpisana przez ustawienia użytkownika.

---

## 4) Zadanie: ulepszenie testu
**Obszar:** `tests/test_settings.py`.

Aktualny test fallbacku legacy obejmuje przypadek, gdy istnieje tylko `debug_eeg_file=True`.
Brakuje testu konfliktowego, gdy jednocześnie występują oba pola (`debug_logging_enabled` i `debug_eeg_file`) z różnymi wartościami.

**Propozycja:**
- dodać test parametryzowany dla kombinacji konfliktowych,
- potwierdzić, że zawsze wygrywa nowe pole `debug_logging_enabled`,
- dodać asercję regresyjną dla przypadku braku obu pól.
