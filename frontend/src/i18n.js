export const LANGUAGE_STORAGE_KEY = "gamweaver-language";
export const DEFAULT_LANGUAGE = "en";

const translations = {
  en: {
    "language.switchToEnglish": "Switch language to English",
    "language.switchToGerman": "Switch language to German",

    "common.yes": "Yes",
    "common.no": "No",
    "common.cancel": "Cancel",
    "common.username": "Username",
    "common.password": "Password",
    "common.loading": "Loading...",
    "common.refresh": "Refresh",

    "header.subtitle": "Interactive GAM Editor",
    "header.modelTrained": "Model Trained",
    "header.modelNotTrained": "Model Not Trained",
    "header.recordsLoaded": "{count} records loaded",

    "login.title": "The Interactive GAM Editor",
    "login.subtitleLogin": "Sign in with your credentials",
    "login.subtitleRegister": "Register with an invite link",
    "login.usernamePlaceholder": "Your username",
    "login.passwordPlaceholder": "Your password",
    "login.registerInviteOnly": "Registration is only possible via an invite link.",
    "login.error.enterCredentials": "Please enter username and password",
    "login.error.inviteRequired": "Registration requires a valid invite link",
    "login.error.loginFailed": "Failed to login",
    "login.error.registerFailed": "Failed to register",
    "login.submitSigningIn": "Signing in...",
    "login.submitRegistering": "Registering...",
    "login.submitSignIn": "Sign In",
    "login.submitRegister": "Register",

    "app.loggedInAs": "Logged in as",
    "app.viewCombinedResults": "View Combined Results",
    "app.superadmin": "Superadmin",
    "app.logout": "Logout",
    "app.footerDescription":
      "GAMWeaver - Interactive GAM Editor for interactive modeling and explainability. Developed by {name} as part of a master's thesis.",
    "app.githubLink": "GAMWeaver GitHub",
    "app.notification.editRemoved": "Edit Removed",
    "app.notification.editsRemoved": "Edits Removed",
    "app.notification.singleMessage":
      "One of your edits was removed by another user:",
    "app.notification.multipleMessage":
      "{count} of your edits were removed by other users:",
    "app.notification.feature": "Feature",
    "app.notification.by": "by",
    "app.notification.xValue": "X Value",
    "app.notification.reason": "Reason",
    "app.notification.gotIt": "Got it",

    "training.title": "Model Training",
    "training.restrictedNotice":
      "Dataset upload/loading and training controls are restricted to the superadmin.",
    "training.dataLoaded": "Data loaded",
    "training.modelTrainedLabel": "Model trained",
    "training.activeDataset": "Active dataset",
    "training.stepLoadData": "Load Data",
    "training.stepLoaded": "Loaded",
    "training.loadDescription":
      "Upload a CSV dataset, select the prediction target, and choose which columns to import.",
    "training.uploadAndLoad": "Upload & Load Dataset",
    "training.stepTrainModel": "Train Model",
    "training.stepTrained": "Trained",
    "training.trainDescription": "Configure and train the IGANN model.",
    "training.estimators": "Number of Estimators",
    "training.estimatorHint":
      "More estimators = better accuracy but slower training",
    "training.trainButton": "Train IGANN Model",
    "training.uploadModalTitle": "Upload Dataset",
    "training.csvFile": "CSV File",
    "training.uploadInspect": "Upload & Inspect Columns",
    "training.columnsDetected": "Columns detected",
    "training.targetColumn": "Target Column",
    "training.featureColumns": "Feature Columns",
    "training.selectAll": "Select all",
    "training.clearAll": "Clear all",
    "training.target": "Target",
    "training.selectedFeatures": "Selected features",
    "training.loadDataset": "Load Dataset",
    "training.error.chooseCsvFirst": "Please choose a CSV file first.",
    "training.error.uploadFailed": "Failed to upload dataset",
    "training.error.loadFailed": "Failed to load dataset",
    "training.error.uploadNeedsSelections":
      "Please upload a CSV, choose a target column, and select at least one feature column.",
    "training.progress.loadingData": "Loading data...",
    "training.progress.training":
      "Training model... This may take a moment.",
    "training.uploading": "Uploading...",
    "training.loadingDataset": "Loading...",
    "training.training": "Training...",

    "dataSummary.title": "Data Summary",
    "dataSummary.empty": "Load the data to see summary statistics",
    "dataSummary.totalRecords": "Total Records",
    "dataSummary.targetVariable": "Target Variable",
    "dataSummary.targetFallback": "Target",
    "dataSummary.mean": "Mean",
    "dataSummary.stdDev": "Std Dev",
    "dataSummary.min": "Min",
    "dataSummary.max": "Max",
    "dataSummary.features": "Features",

    "prediction.title": "Make Prediction",
    "prediction.description":
      "Set dataset feature values and run a prediction. Inputs are generated from the active dataset schema.",
    "prediction.trainFirst":
      "Please train the model first before making predictions.",
    "prediction.noSchema":
      "No feature schema available. Ask the superadmin to load a dataset.",
    "prediction.predicting": "Predicting...",
    "prediction.predictTarget": "Predict {target}",
    "prediction.predictedTarget": "Predicted {target}",

    "superadmin.error.loadUsers": "Failed to load users",
    "superadmin.error.usernamePasswordRequired":
      "Username and password are required",
    "superadmin.error.createUser": "Failed to create user",
    "superadmin.error.createInvite": "Failed to create invite",
    "superadmin.error.resetDatabase": "Failed to reset database",
    "superadmin.title": "Superadmin Overview",
    "superadmin.openCombined": "Open Combined Results",
    "superadmin.createUser": "Create User",
    "superadmin.inviteLink": "Invite Link",
    "superadmin.generateInvite": "Generate Invite",
    "superadmin.inviteCopied": "Invite link copied to clipboard.",
    "superadmin.expires": "Expires",
    "superadmin.inviteDescription":
      "Generate an invite link to allow new registrations.",
    "superadmin.systemReset": "System Reset",
    "superadmin.systemResetDescription":
      "Permanently removes all users, all saved edits, uploaded CSV files, and extracted feature state for the current backend environment.",
    "superadmin.resetDatabase": "Reset Database",
    "superadmin.creating": "Creating...",
    "superadmin.users": "Users",
    "superadmin.loadingUsers": "Loading users...",
    "superadmin.noUsers": "No users yet.",
    "superadmin.created": "Created",
    "superadmin.badge": "Superadmin",
    "superadmin.resetConfirmTitle": "Reset Database?",
    "superadmin.resetConfirmBody":
      "This permanently deletes users, edits, uploaded CSV files, and extracted feature state. This action cannot be undone.",
    "superadmin.resetting": "Resetting...",
    "superadmin.resetEverything": "Yes, Reset Everything",
  },
  de: {
    "language.switchToEnglish": "Sprache auf Englisch umstellen",
    "language.switchToGerman": "Sprache auf Deutsch umstellen",

    "common.yes": "Ja",
    "common.no": "Nein",
    "common.cancel": "Abbrechen",
    "common.username": "Benutzername",
    "common.password": "Passwort",
    "common.loading": "Laden...",
    "common.refresh": "Aktualisieren",

    "header.subtitle": "Interaktiver GAM-Editor",
    "header.modelTrained": "Modell trainiert",
    "header.modelNotTrained": "Modell nicht trainiert",
    "header.recordsLoaded": "{count} Datensaetze geladen",

    "login.title": "Der interaktive GAM-Editor",
    "login.subtitleLogin": "Mit deinen Zugangsdaten anmelden",
    "login.subtitleRegister": "Mit einem Einladungslink registrieren",
    "login.usernamePlaceholder": "Dein Benutzername",
    "login.passwordPlaceholder": "Dein Passwort",
    "login.registerInviteOnly":
      "Eine Registrierung ist nur ueber einen Einladungslink moeglich.",
    "login.error.enterCredentials":
      "Bitte Benutzername und Passwort eingeben",
    "login.error.inviteRequired":
      "Die Registrierung erfordert einen gueltigen Einladungslink",
    "login.error.loginFailed": "Anmeldung fehlgeschlagen",
    "login.error.registerFailed": "Registrierung fehlgeschlagen",
    "login.submitSigningIn": "Anmeldung laeuft...",
    "login.submitRegistering": "Registrierung laeuft...",
    "login.submitSignIn": "Anmelden",
    "login.submitRegister": "Registrieren",

    "app.loggedInAs": "Angemeldet als",
    "app.viewCombinedResults": "Kombinierte Ergebnisse anzeigen",
    "app.superadmin": "Superadmin",
    "app.logout": "Abmelden",
    "app.footerDescription":
      "GAMWeaver - Interaktiver GAM-Editor fuer interaktive Modellierung und Erklaerbarkeit. Entwickelt von {name} im Rahmen einer Masterarbeit.",
    "app.githubLink": "GAMWeaver GitHub",
    "app.notification.editRemoved": "Bearbeitung entfernt",
    "app.notification.editsRemoved": "Bearbeitungen entfernt",
    "app.notification.singleMessage":
      "Eine deiner Bearbeitungen wurde von einer anderen Person entfernt:",
    "app.notification.multipleMessage":
      "{count} deiner Bearbeitungen wurden von anderen Personen entfernt:",
    "app.notification.feature": "Merkmal",
    "app.notification.by": "von",
    "app.notification.xValue": "X-Wert",
    "app.notification.reason": "Grund",
    "app.notification.gotIt": "Verstanden",

    "training.title": "Modelltraining",
    "training.restrictedNotice":
      "Das Hochladen/Laden von Datensaetzen und die Trainingssteuerung sind auf den Superadmin beschraenkt.",
    "training.dataLoaded": "Daten geladen",
    "training.modelTrainedLabel": "Modell trainiert",
    "training.activeDataset": "Aktiver Datensatz",
    "training.stepLoadData": "Daten laden",
    "training.stepLoaded": "Geladen",
    "training.loadDescription":
      "CSV-Datensatz hochladen, Vorhersageziel auswaehlen und zu importierende Spalten festlegen.",
    "training.uploadAndLoad": "Datensatz hochladen und laden",
    "training.stepTrainModel": "Modell trainieren",
    "training.stepTrained": "Trainiert",
    "training.trainDescription": "IGANN-Modell konfigurieren und trainieren.",
    "training.estimators": "Anzahl der Estimatoren",
    "training.estimatorHint":
      "Mehr Estimatoren = bessere Genauigkeit, aber langsameres Training",
    "training.trainButton": "IGANN-Modell trainieren",
    "training.uploadModalTitle": "Datensatz hochladen",
    "training.csvFile": "CSV-Datei",
    "training.uploadInspect": "Hochladen und Spalten pruefen",
    "training.columnsDetected": "Erkannte Spalten",
    "training.targetColumn": "Zielspalte",
    "training.featureColumns": "Feature-Spalten",
    "training.selectAll": "Alle waehlen",
    "training.clearAll": "Auswahl loeschen",
    "training.target": "Ziel",
    "training.selectedFeatures": "Ausgewaehlte Features",
    "training.loadDataset": "Datensatz laden",
    "training.error.chooseCsvFirst": "Bitte zuerst eine CSV-Datei auswaehlen.",
    "training.error.uploadFailed": "Datensatz konnte nicht hochgeladen werden",
    "training.error.loadFailed": "Datensatz konnte nicht geladen werden",
    "training.error.uploadNeedsSelections":
      "Bitte eine CSV hochladen, eine Zielspalte waehlen und mindestens eine Feature-Spalte auswaehlen.",
    "training.progress.loadingData": "Daten werden geladen...",
    "training.progress.training":
      "Modell wird trainiert... Das kann einen Moment dauern.",
    "training.uploading": "Wird hochgeladen...",
    "training.loadingDataset": "Wird geladen...",
    "training.training": "Training laeuft...",

    "dataSummary.title": "Datenuebersicht",
    "dataSummary.empty": "Daten laden, um Zusammenfassungsstatistiken zu sehen",
    "dataSummary.totalRecords": "Gesamtanzahl Datensaetze",
    "dataSummary.targetVariable": "Zielvariable",
    "dataSummary.targetFallback": "Ziel",
    "dataSummary.mean": "Mittelwert",
    "dataSummary.stdDev": "Std.-Abw.",
    "dataSummary.min": "Min",
    "dataSummary.max": "Max",
    "dataSummary.features": "Features",

    "prediction.title": "Vorhersage erstellen",
    "prediction.description":
      "Feature-Werte des Datensatzes setzen und eine Vorhersage ausfuehren. Die Eingaben werden aus dem aktiven Datensatzschema generiert.",
    "prediction.trainFirst":
      "Bitte zuerst das Modell trainieren, bevor Vorhersagen erstellt werden.",
    "prediction.noSchema":
      "Kein Feature-Schema verfuegbar. Bitte den Superadmin, einen Datensatz zu laden.",
    "prediction.predicting": "Vorhersage laeuft...",
    "prediction.predictTarget": "{target} vorhersagen",
    "prediction.predictedTarget": "Vorhergesagtes {target}",

    "superadmin.error.loadUsers": "Benutzer konnten nicht geladen werden",
    "superadmin.error.usernamePasswordRequired":
      "Benutzername und Passwort sind erforderlich",
    "superadmin.error.createUser":
      "Benutzer konnte nicht erstellt werden",
    "superadmin.error.createInvite":
      "Einladung konnte nicht erstellt werden",
    "superadmin.error.resetDatabase":
      "Datenbank konnte nicht zurueckgesetzt werden",
    "superadmin.title": "Superadmin-Uebersicht",
    "superadmin.openCombined": "Kombinierte Ergebnisse oeffnen",
    "superadmin.createUser": "Benutzer erstellen",
    "superadmin.inviteLink": "Einladungslink",
    "superadmin.generateInvite": "Einladung erzeugen",
    "superadmin.inviteCopied":
      "Einladungslink wurde in die Zwischenablage kopiert.",
    "superadmin.expires": "Laeuft ab",
    "superadmin.inviteDescription":
      "Einen Einladungslink erzeugen, um neue Registrierungen zu ermoeglichen.",
    "superadmin.systemReset": "System zuruecksetzen",
    "superadmin.systemResetDescription":
      "Entfernt dauerhaft alle Benutzer, alle gespeicherten Bearbeitungen, hochgeladene CSV-Dateien und den extrahierten Feature-Zustand der aktuellen Backend-Umgebung.",
    "superadmin.resetDatabase": "Datenbank zuruecksetzen",
    "superadmin.creating": "Wird erstellt...",
    "superadmin.users": "Benutzer",
    "superadmin.loadingUsers": "Benutzer werden geladen...",
    "superadmin.noUsers": "Noch keine Benutzer.",
    "superadmin.created": "Erstellt",
    "superadmin.badge": "Superadmin",
    "superadmin.resetConfirmTitle": "Datenbank zuruecksetzen?",
    "superadmin.resetConfirmBody":
      "Dadurch werden Benutzer, Bearbeitungen, hochgeladene CSV-Dateien und extrahierter Feature-Zustand dauerhaft geloescht. Diese Aktion kann nicht rueckgaengig gemacht werden.",
    "superadmin.resetting": "Zuruecksetzen laeuft...",
    "superadmin.resetEverything": "Ja, alles zuruecksetzen",
  },
};

export const getInitialLanguage = () => {
  if (typeof window === "undefined") return DEFAULT_LANGUAGE;
  const storedLanguage = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
  return storedLanguage === "de" ? "de" : DEFAULT_LANGUAGE;
};

export const getNumberLocale = (language) =>
  language === "de" ? "de-DE" : "en-US";

export const getDateLocale = (language) =>
  language === "de" ? "de-DE" : "en-US";

export const createTranslator = (language) => (key, variables = {}) => {
  const template =
    translations[language]?.[key] ?? translations[DEFAULT_LANGUAGE]?.[key] ?? key;

  return template.replace(/\{(\w+)\}/g, (_, variableName) => {
    const value = variables[variableName];
    return value === undefined || value === null ? "" : String(value);
  });
};
