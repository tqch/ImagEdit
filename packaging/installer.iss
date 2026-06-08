; Inno Setup script for the ImagEdit Windows installer.
; Compile (after the PyInstaller build) with:
;   ISCC.exe /DMyAppVersion=0.1.0 packaging\installer.iss
; CI passes the version from the git tag automatically.

#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif

[Setup]
AppId={{8E1FAE2B-6E51-4C7E-9C7A-1B6C2D9D4A21}
AppName=ImagEdit
AppVersion={#MyAppVersion}
AppPublisher=Tianqi Chen
DefaultDirName={autopf}\ImagEdit
DefaultGroupName=ImagEdit
UninstallDisplayIcon={app}\ImagEdit.exe
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
OutputDir=..\dist
OutputBaseFilename=ImagEdit-Setup-{#MyAppVersion}
SetupIconFile=..\imagedit\assets\icon.ico
; Show the GPLv3 license in the installer (user must accept).
LicenseFile=..\LICENSE
; Per-user install: no admin rights needed (installs under %LocalAppData%).
PrivilegesRequired=lowest

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; \
    GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "..\dist\ImagEdit\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\THIRD-PARTY-NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\ImagEdit"; Filename: "{app}\ImagEdit.exe"
Name: "{autodesktop}\ImagEdit"; Filename: "{app}\ImagEdit.exe"; \
    Tasks: desktopicon

[Run]
Filename: "{app}\ImagEdit.exe"; Description: "Launch ImagEdit"; \
    Flags: nowait postinstall skipifsilent
