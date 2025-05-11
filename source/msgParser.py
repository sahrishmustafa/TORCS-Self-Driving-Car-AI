class MsgParser(object):
    '''
    A parser for received UDP messages and building UDP messages
    '''
    def __init__(self):
        '''Constructor'''
        
    def parse(self, str_sensors):
        '''Return a dictionary with tags and values from the UDP message'''
        sensors = {}
        
        b_open = str_sensors.find('(')
        
        while b_open >= 0:
            b_close = str_sensors.find(')', b_open)
            if b_close >= 0:
                substr = str_sensors[b_open + 1: b_close]
                items = substr.split()
                if len(items) < 2:
                    print ("Problem parsing substring: ", substr)
                else:
                    value = []
                    for i in range(1,len(items)):
                        value.append(items[i])
                    sensors[items[0]] = value
                b_open = str_sensors.find('(', b_close)
            else:
                print ("Problem parsing sensor string: ", str_sensors)
                return None
        
        return sensors
    
    def stringify(self, dictionary):
        '''Build an UDP message from a dictionary'''
        msg = ''
        
        for key, value in dictionary.items():
            if value != None and value[0] != None:
                msg += '(' + key
                for val in value:
                    msg += ' ' + str(val)
                msg += ')'
        
        return msg
    
    def dict_to_msg(self, data_dict):
        """
        Converts a dictionary like {'accel': 0.123, 'gear': 3} into a TORCS-style message string.
        """
        msg_parts = []
        required_keys = ['accel', 'brake', 'gear', 'steer', 'focus', 'meta']
        for key in required_keys:
            value = data_dict.get(key, 0)  # Default to 0 if missing
            if isinstance(value, list):
                val_str = ' '.join(str(v) for v in value)
                msg_parts.append(f"({key} {val_str})")
            else:
                msg_parts.append(f"({key} {value})")
    
        return ''.join(msg_parts)
