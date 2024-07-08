//#include "SerialControl.h"
#include "pch.h"

SerialPortControl::SerialPortControl(unsigned int unUARTNo /*= 1*/, double dBauteRate /*=115200*/, unsigned int unDataBits /*= 8u*/, unsigned int unStopBits /*= 1u*/, char cParity /*= 'N'*/) {
	m_unUARTNo = unUARTNo;
	m_dBauteRate = dBauteRate;
	m_unDataBits = unDataBits;
	m_unStopBits = unStopBits;
	m_cParity = cParity;
	m_nReceiveLength = 0;
	m_pcReceiveData = NULL;
	m_hComDevice = INVALID_HANDLE_VALUE;
}

SerialPortControl::SerialPortControl() {
	m_unUARTNo = 1;
	m_dBauteRate = 115200;
	m_unDataBits = 8u;
	m_unStopBits = 1u;
	m_cParity = 'N';
	m_nReceiveLength = 0;
	m_pcReceiveData = NULL;
	m_hComDevice = INVALID_HANDLE_VALUE;
}

int SerialPortControl::InitPort() {
	char pcComParam[50];
	sprintf_s(pcComParam, "baud=%g parity=%c data=%u stop=%u", m_dBauteRate, m_cParity, m_unDataBits, m_unStopBits);

	if (!OpenPort()) {
		return 0;
	}
	if (SetupComm(m_hComDevice, 10*1024, 10*1024)) {
		COMMTIMEOUTS ctoData;
		ctoData.ReadIntervalTimeout = 500;
		ctoData.ReadTotalTimeoutMultiplier = 500;
		ctoData.ReadTotalTimeoutConstant = 500;
		ctoData.WriteTotalTimeoutMultiplier = 500;
		ctoData.WriteTotalTimeoutConstant = 500;

		if (SetCommTimeouts(m_hComDevice, &ctoData)) {
			DCB dcbCurrent;

			DWORD dwNum = MultiByteToWideChar(CP_ACP, 0, pcComParam, -1, NULL, 0);
			wchar_t* pwText = new wchar_t[dwNum];
			MultiByteToWideChar(CP_ACP, 0, pcComParam, -1, pwText, dwNum);
			GetCommState(m_hComDevice, &dcbCurrent);
			if (BuildCommDCB(pwText, &dcbCurrent)) {
				dcbCurrent.fRtsControl = RTS_CONTROL_ENABLE;
				delete[] pwText;

				if (SetCommState(m_hComDevice, &dcbCurrent)) {
					PurgeComm(m_hComDevice, PURGE_RXCLEAR | PURGE_TXCLEAR | PURGE_RXABORT | PURGE_TXABORT);
				}
				else {
					return 0;
				}
			}
			else {
				delete[] pwText;
				return 0;
			}
		}
		else {
			return 0;
		}

	}
	else {
		return 0;
	}
	return 1;
}

int SerialPortControl::OpenPort() {
	//if (!FindPort()) {
	//	return 0;
	//}



	char cTempPortName[50];
	sprintf_s(cTempPortName, "COM%d", m_unUARTNo);

	m_hComDevice = CreateFileA(cTempPortName, GENERIC_READ | GENERIC_WRITE,
		0,
		NULL,
		OPEN_EXISTING,
		0,
		0
	);

	if (m_hComDevice == INVALID_HANDLE_VALUE) {
		return 0;
	}

	return 1;	
}

void SerialPortControl::ClosePort() {
	if (m_hComDevice != INVALID_HANDLE_VALUE) {
		CloseHandle(m_hComDevice);
		m_hComDevice = INVALID_HANDLE_VALUE;
	}
}

int SerialPortControl::SendData(unsigned char* pucSendData, unsigned int unLength) {
	if (m_hComDevice == INVALID_HANDLE_VALUE) {
		return 0;
	}

	DWORD dwByteToSend = 0ul;

	bool bResult = WriteFile(m_hComDevice, pucSendData, unLength, &dwByteToSend, NULL);

	if (!bResult) {

		PurgeComm(m_hComDevice, PURGE_RXCLEAR | PURGE_RXABORT);
		return false;
	}

	return dwByteToSend;
}

int SerialPortControl::SendData(char* pcSendData, unsigned int unLength) {
	if (m_hComDevice == INVALID_HANDLE_VALUE) {
		return 0;
	}

	DWORD dwByteToSend = 0ul;

	bool bResult = WriteFile(m_hComDevice, pcSendData, unLength, &dwByteToSend, NULL);

	if (!bResult) {

		PurgeComm(m_hComDevice, PURGE_RXCLEAR | PURGE_RXABORT);
		return false;
	}

	return dwByteToSend;
}

unsigned int SerialPortControl::GetBytesInBuffer() {
	COMSTAT comstat;

	memset(&comstat, 0, sizeof(COMSTAT));

	unsigned int unBytesInQue = 0u;
	DWORD dwErrorCode;

	if (ClearCommError(m_hComDevice, &dwErrorCode, &comstat)) {
		unBytesInQue = comstat.cbInQue;
	}

	return unBytesInQue;
}

int SerialPortControl::ReceiveData() {
	if (m_hComDevice == INVALID_HANDLE_VALUE) {
		return 0;
	}
	unsigned int unReceiveBufferSz = GetBytesInBuffer();

	if (m_pcReceiveData) {
		delete[] m_pcReceiveData;
		m_pcReceiveData = NULL;
	}

	DWORD dwReceiveDataSz = 0ul;

	if (unReceiveBufferSz) {
		m_pcReceiveData = new char[unReceiveBufferSz+1];
		//char* cReceiveData = new char[256];


		if (ReadFile(m_hComDevice, m_pcReceiveData, unReceiveBufferSz, &dwReceiveDataSz, NULL)) {
			m_nReceiveLength = dwReceiveDataSz;
			PurgeComm(m_hComDevice, PURGE_RXCLEAR|PURGE_RXABORT);
			return dwReceiveDataSz;
		}
		else {
			return 0;
		}
	}

	return 0;


}

int SerialPortControl::FindPort() {

	char cTempComName[50];
	HANDLE hTemp = INVALID_HANDLE_VALUE;

	for (int n = 0; n <= cnMaxComSize; n++) {
		memset(cTempComName, 0, 50*sizeof(char));
		sprintf_s(cTempComName, "COM%d", n);
		hTemp = INVALID_HANDLE_VALUE;
		hTemp = CreateFileA(cTempComName, GENERIC_READ | GENERIC_WRITE,
			0,
			NULL,
			OPEN_EXISTING,
			0,
			0
		);
		if (hTemp != INVALID_HANDLE_VALUE) {
			m_unUARTNo = n;
			CloseHandle(hTemp);
			return 1;
		}

	}
	return 0;
}

int SerialPortControl::IsPortAlive(unsigned int unUARTNo) {
	
	char cTempComName[50];
	sprintf_s(cTempComName, "COM%ud", unUARTNo);

	HANDLE hTemp = CreateFileA(cTempComName, GENERIC_READ | GENERIC_WRITE,  
							0,                          
							NULL,                       
							OPEN_EXISTING,              
							0,
							0
							);
	if (hTemp == INVALID_HANDLE_VALUE) {
		return 0;
	}
	return 1;
}

SerialPortControl::~SerialPortControl() {
	if (m_hComDevice != INVALID_HANDLE_VALUE) {
		CloseHandle(m_hComDevice);
	}

	if (m_pcReceiveData) {
		delete[] m_pcReceiveData;
		m_pcReceiveData = NULL;
	}
}

char*& SerialPortControl::GetReceiveData() {
	return m_pcReceiveData;
}

int SerialPortControl::GetReceiveLength() {
	return m_nReceiveLength;
}
