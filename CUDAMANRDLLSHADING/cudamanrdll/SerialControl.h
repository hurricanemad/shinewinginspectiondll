#pragma once



const int cnMaxComSize = 256;


class SerialPortControl {
private:
	unsigned int m_unUARTNo;
	double m_dBauteRate;
	unsigned int m_unDataBits;
	unsigned int m_unStopBits;
	char m_cParity;
	char* m_pcReceiveData;
	int m_nReceiveLength;
	HANDLE m_hComDevice;
	bool m_bSerialPortInitSuccess;

public:
	SerialPortControl(unsigned int unUARTNo, double dBauteRate = 115200, unsigned int unDataBits = 8u, unsigned int unStopBits = 1u, char cParity = 'N');
	SerialPortControl();

	int InitPort();
	int OpenPort();
	void ClosePort();
	int SendData(unsigned char* pucSendData, unsigned int unLength);
	int SendData(char* pcSendData, unsigned int unLength);
	int ReceiveData();
	int FindPort();
	int IsPortAlive(unsigned int unUARTNo);
	unsigned int GetBytesInBuffer();
	bool GetSerialPortInitSuccess();

	int GetUARTNo() {
		return m_unUARTNo;
	}

	void SetUARTNo(int unUARTNo) {
		m_unUARTNo = unUARTNo;
	}

	char*& GetReceiveData();
	int GetReceiveLength();

	ofstream foutSerialPort;
	//int m_nTestLog;

	~SerialPortControl();

};