/* eslint-disable @typescript-eslint/no-empty-object-type */
export interface CommonResponse<T> {
  customMessage?: string;
  message?: string;
  data: T;
  code?: number;
  subCode?: string;
}

export interface EmptyResponse extends CommonResponse<any> {}

export interface ListResponse<T> extends CommonResponse<{ list: T[] }> {}

export interface PagingResponse<T>
  extends CommonResponse<{
    list: T[];
    total: number;
  }> {}
